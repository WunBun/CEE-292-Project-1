# %%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    TensorDataset,
    random_split,
    DataLoader,
    BatchSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.profiler import profile, ProfilerActivity

from tqdm import tqdm

import copy

from collections import defaultdict

import time

plt.rcParams["axes3d.mouserotationstyle"] = "azel"

# %%


class TensorBatches:  # a fast batched data loader that avoids torch overhead
    def __init__(self, ds, batch_size, shuffle=False):
        if isinstance(ds, torch.utils.data.Subset):
            self.tensors = [t[ds.indices] for t in ds.dataset.tensors]
        else:
            self.tensors = list(ds.tensors)
        self.n, self.bs, self.shuffle = len(self.tensors[0]), batch_size, shuffle

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        idx = torch.randperm(self.n) if self.shuffle else None
        for s in range(0, self.n, self.bs):
            if idx is None:
                yield tuple(t[s : s + self.bs] for t in self.tensors)
            else:
                b = idx[s : s + self.bs]
                yield tuple(t[b] for t in self.tensors)


class FDMSolve(
    torch.autograd.Function
):  # adjunct for derivative of FDM solve for faster training
    """
    Solves Dn(q) X = rhs with Dn assembled sparsely from q.
    The backward pass computes only the entries of dL/dDn that q touches,
    instead of the full outer product.

    Not in use yet lmao
    """

    @staticmethod
    def forward(ctx, q_vecs, rhs, mesh):
        B, F = q_vecs.shape[0], mesh.n_free

        vals = q_vecs[:, mesh.Dn_edge] * mesh.Dn_sign
        Dn = torch.zeros(B, F * F, dtype=q_vecs.dtype, device=q_vecs.device)
        Dn.index_add_(1, mesh.Dn_flat, vals)

        LU, piv, info = torch.linalg.lu_factor_ex(Dn.view(B, F, F))
        X = torch.linalg.lu_solve(LU, piv, rhs)

        ctx.save_for_backward(LU, piv, X)
        ctx.mesh = mesh
        return X

    @staticmethod
    def backward(ctx, gX):
        LU, piv, X = ctx.saved_tensors
        m = ctx.mesh

        gB = torch.linalg.lu_solve(LU, piv, gX, adjoint=True)
        g_vals = -(gB[:, m.Dn_i, :] * X[:, m.Dn_j, :]).sum(-1)
        g_q = torch.zeros(gX.shape[0], m.n_edges, dtype=gX.dtype, device=gX.device)
        g_q.index_add_(1, m.Dn_edge, g_vals * m.Dn_sign)

        return g_q, gB, None


def setdiff1d_t(t1, t2):  # helper function
    mask = ~torch.isin(t1, t2)
    return t1[mask]


class MeshData:
    """
    Class containing mesh information for FDM
        - n_nodes: Number of nodes
        - n_edges: Number of edges
        - C: Connectivity matrix
        - X: Original node locations
        - free_ind: indices of free nodes
        - fixed_ind: indices of fixed nodes
    """

    def __init__(self, n=3, x_space=0.5, y_space=0.5):
        """
        - Generate the connectivity + data for a shell on an nxn grid.
        - All of the edges around the border are supports.
        - Units are m and kN.

        inputs:
            - n: side length of the generated square mesh
            - x_space: distance (in m) between nodes along the x-axis
            - y_space: distance (in m) between nodes along the y-axis

        outputs:
            - n_nodes: number of nodes in the mesh
            - n_edges: number of edges in the mesh
            - C: n_edges x n_nodes connectivity matrix
                If there is an edge i from node k to node m, then row i contains
                -1 in column k and 1 in column m. The graph is undirected, so the
                choice of which column to assign to -1 is arbitrary.
            - X: n_nodes x 3 original node positions matrix
            - free_ind: vector of indices of free nodes
            - fixed_ind: vector of indices of fixed nodes
        """

        self.n_nodes = n**2
        self.n_edges = 2 * (n - 1) * n

        # create mesh

        xs = torch.linspace(0, x_space * (n - 1), n)
        ys = torch.linspace(0, y_space * (n - 1), n)

        xv, yv = torch.meshgrid(xs, ys, indexing="ij")

        # original position vector
        # save non-support positions as well so we can preview the mesh
        self.X = torch.column_stack((xv.ravel(), yv.ravel(), torch.zeros(n**2)))

        # create connectivity matrix
        self.C = torch.zeros((self.n_edges, self.n_nodes))

        edge_ctr = 0
        for node_ind in range(self.n_nodes):
            # if node_ind + 1 is a multiple of n, node_ind is at the end of a row
            # and doesn't have a neighbor to the right so we skip

            if (node_ind + 1) % n != 0 and node_ind + 1 < self.n_nodes:
                # print(f"Adding edge {edge_ctr} from node {node_ind}")
                self.C[edge_ctr, node_ind] = -1
                self.C[edge_ctr, node_ind + 1] = 1
                edge_ctr += 1

            # if the ind of the node above is out of range, skip

            if node_ind + n < self.n_nodes:
                # print(f"Adding edge {edge_ctr} from node {node_ind}")
                self.C[edge_ctr, node_ind] = -1
                self.C[edge_ctr, node_ind + n] = 1
                edge_ctr += 1

        # fixed indices are all of those along the outer boundary
        fixed_ind = [
            ind
            for ind in range(self.n_nodes)
            if (
                (ind - n < 0)
                or (ind + n >= self.n_nodes)
                or (ind % n == 0)
                or ((ind + 1) % n == 0)
            )
        ]

        self.fixed_ind = torch.tensor(fixed_ind)
        self.n_fixed = len(self.fixed_ind)

        all_ind = torch.tensor(range(self.n_nodes))
        self.free_ind = setdiff1d_t(all_ind, self.fixed_ind)
        self.n_free = len(self.free_ind)

        self.is_fixed = torch.zeros(self.n_nodes)
        self.is_fixed[self.fixed_ind] = 1.0

        self.Cn = self.C[:, self.free_ind]
        self.Cf = self.C[:, self.fixed_ind]

        self.Xf = self.X[self.fixed_ind, :]

        # precompute for batched solver
        self.CnT = self.Cn.T.contiguous()  # n_free x n_edges
        self.CfXf = self.Cf @ self.Xf  # n_edges x 3

        outer = self.Cn.unsqueeze(2) * self.Cn.unsqueeze(1)  # n_edges x n_free x n_free
        e_idx, i_idx, j_idx = outer.nonzero(as_tuple=True)
        self.Dn_edge = e_idx  # which edge each entry comes from
        self.Dn_flat = i_idx * self.n_free + j_idx  # where it lands in flattened Dn
        self.Dn_sign = outer[e_idx, i_idx, j_idx]  # +1 or -1
        self.Dn_i = i_idx
        self.Dn_j = j_idx

        M = self.CnT.unsqueeze(2) * self.CfXf.unsqueeze(0)
        M = M.permute(1, 0, 2).reshape(self.n_edges, self.n_free * 3)
        self.bnd_edges = M.abs().sum(1).nonzero().squeeze(1)  # boundary edges
        self.rhs_map = M[self.bnd_edges].contiguous()

    def eq_shape(self, q_vec, p_mat):
        """
        The governing equation of the force density method is
        C_n.T @ Q @ C_n @ X_n + C_n.T @ Q @ C_f @ X_f + p = 0.

        Solves for the equilibrium shape given an input force density
        vector and load matrix.

        Positive force densities represent compression.

        Differentiable by torch.

        inputs:
            - mesh_data: dictionary containing mesh data from
                generate_mesh_and_connectivity()
                * Derivatives wrt mesh_data not required
                    because it doesn't change
                    (I think)

            - q_vec: 1 x n_edges tensor containing force densities.
                Requires grad

            - p_mat: n_free x 3 tensor containing nodal loads.
                Doesn't require grad bc loading is imposed, not solved for

        outputs:
            - X_eq
        """

        Q = torch.diag(q_vec)

        # print(f"{self.Cn=}")
        # print(f"{Q=}")

        Dn = self.Cn.T @ Q @ self.Cn
        Df = self.Cn.T @ Q @ self.Cf

        X_free_eq = torch.linalg.solve(Dn, -p_mat - Df @ self.Xf)

        X_eq = self.X.clone().detach()
        X_eq[self.free_ind, :] = X_free_eq

        return X_eq

    def eq_shape_batched(
        self,
        q_vecs,
        p_mats,
    ):
        """
        Batched version of eq_shape.

        inputs:
            - q_vecs: n_batches x n_edges tensor of force densities
            - p_mats: n_batches x n_free x 3 nodal loads
        outputs:
            - X_eq: n_batches x n-nodes x 3 tensor of equilibrium nodal positions

        Uses Cn.T @ diag(q) = Cn.T * q (column scaling) so no dense
        force density matrix is built. All the linear systems
        are solved in a single call yay
        """

        n_batches = q_vecs.shape[0]

        vals = q_vecs[:, self.Dn_edge] * self.Dn_sign
        Dn = torch.zeros(
            n_batches, self.n_free**2, dtype=q_vecs.dtype, device=q_vecs.device
        )
        Dn.index_add_(1, self.Dn_flat, vals)
        Dn = Dn.view(n_batches, self.n_free, self.n_free)

        rhs = -p_mats - (q_vecs[:, self.bnd_edges] @ self.rhs_map).view(
            n_batches, self.n_free, 3
        )  # n_batches x n_free x 3

        X_free_eq = torch.linalg.solve(Dn, rhs)  # n_batches x n_free x 3

        # X_free_eq = FDMSolve.apply(q_vecs, rhs, self)

        X_eq = self.X.unsqueeze(0).expand(n_batches, -1, -1).clone()
        X_eq[:, self.free_ind, :] = X_free_eq

        return X_eq


# %%


def visualize_mesh(mesh_data, X_eq=None, hat_X=None):
    """
    Plots the mesh from mesh_data
    """

    if X_eq is not None:
        X = X_eq.detach().numpy()
    else:
        X = mesh_data.X.numpy()

    n_edges = mesh_data.n_edges
    C = mesh_data.C.numpy()
    n = int(np.sqrt(mesh_data.n_nodes))

    edge_inds = [np.where(C[i, :])[0] for i in range(n_edges)]

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 6), dpi=100, subplot_kw={"projection": "3d"}
    )

    segments = [X[edge, :] for edge in edge_inds]

    lc = Line3DCollection(
        segments,
        colors="blue",
        linewidths=0.5,
    )

    ax.add_collection(lc)

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            faces.append(
                (i * n + j, (i + 1) * n + j, (i + 1) * n + (j + 1), i * n + (j + 1))
            )

    face_locs = [X[face, :] for face in faces]

    mesh = Poly3DCollection(
        face_locs,
        facecolor=(0.3, 0.5, 0.8, 0.3),
        edgecolor="none",
    )

    ax.add_collection(mesh)

    nodes = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        s=10,
        c="black",
    )

    if hat_X is not None:
        hat_X = hat_X.clone().detach().numpy()

        err = np.linalg.norm((hat_X - X), axis=1)
        max_err_node = np.argmax(err)
        node_sizes = [30 if i == max_err_node else 10 for i in range(hat_X.shape[0])]
        node_colors = [
            "orange" if i == max_err_node else "red" for i in range(hat_X.shape[0])
        ]

        hat_segments = [hat_X[edge, :] for edge in edge_inds]

        hat_lc = Line3DCollection(
            hat_segments,
            colors=("red", 0.5),
            linewidths=0.5,
        )

        ax.add_collection(hat_lc)

        hat_face_locs = [hat_X[face, :] for face in faces]

        # hat_mesh = Poly3DCollection(
        #     hat_face_locs,
        #     facecolor=(0.8, 0.5, 0.3, 0.3),
        #     edgecolor="none",
        # )

        # ax.add_collection(hat_mesh)

        nodes = ax.scatter(
            hat_X[:, 0],
            hat_X[:, 1],
            hat_X[:, 2],
            s=node_sizes,
            c=node_colors,
        )

    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


# %%


class FixedTopologySampleGenerator:
    """
    - Takes in a specific topology, represented by a MeshData object.
    - Generates samples with varying or constant load and force densities.
    - Each node has feature vector [x_goal, y_goal, z_goal, is_fixed, p_x, p_y, p_z]
        Feature vectors are concatenated into a 1 x (n_nodes * 7) tensor
    """

    def __init__(
        self,
        mesh_data: MeshData,
        randomize_q=True,
        randomize_p=False,
        q_range=(0.5, 30),  # range of q-values. positive for now...
        p_range=(
            (0, 0),  # range of x loads
            (0, 0),  # range of y loads
            (-4, 0),  # range of z loads
        ),
    ):
        self.topology = mesh_data
        self.randomize_q = randomize_q
        self.randomize_p = randomize_p

        self.q_range = q_range
        self.p_range = p_range

    def sample(  # legacy, uses a loop rather than batching
        self,
        n=2,
        q_vec=None,
        p_mat=None,
    ):
        """
        Create training data: nodal feature vectors and force densities for
        randomized shells on the same topology.
        """

        samples = []
        qs = []

        for i in range(n):
            if self.randomize_q or (q_vec is None):
                # generate random force densities

                q_vec = (
                    torch.rand(self.topology.n_edges)
                    * (self.q_range[1] - self.q_range[0])
                    + self.q_range[0]
                )

            qs.append(q_vec)

            if self.randomize_p or (p_mat is None):
                # generate random loads

                px_vec = (
                    torch.rand(self.topology.n_free)
                    * (self.p_range[0][1] - self.p_range[0][0])
                    + self.p_range[0][0]
                )

                py_vec = (
                    torch.rand(self.topology.n_free)
                    * (self.p_range[1][1] - self.p_range[1][0])
                    + self.p_range[1][0]
                )

                pz_vec = (
                    torch.rand(self.topology.n_free)
                    * (self.p_range[2][1] - self.p_range[2][0])
                    + self.p_range[2][0]
                )

                p_mat = torch.column_stack((px_vec, py_vec, pz_vec))

            # print(f"{q_vec=}")

            # print(f"{p_mat=}")

            X_eq = self.topology.eq_shape(
                q_vec,
                p_mat,
            )

            X_eq = X_eq

            # print(f"{X_eq=}")

            sample_features = torch.column_stack(
                (X_eq, self.topology.is_fixed, torch.zeros((self.topology.n_nodes, 3)))
            )

            sample_features[self.topology.free_ind, 4:] = p_mat

            # print(f"{sample_features=}")

            samples.append(sample_features)

        sample_tensor = torch.stack(samples)  # n_samples x n_nodes x 7 (# features)
        q_tensor = torch.stack(qs)  # n_samples x n_edges

        return sample_tensor, q_tensor

    def sample_batch(
        self,
        n=2,
        q_vecs=None,
        p_mats=None,
    ):
        """
        Create training data: nodal feature vectors and force densities for
        randomized shells on the same topology.
        """

        if self.randomize_q or (q_vecs is None):
            # generate random force densities

            q_vecs = (
                torch.rand((n, self.topology.n_edges))
                * (self.q_range[1] - self.q_range[0])
                + self.q_range[0]
            )

        if self.randomize_p or (p_mats is None):
            # generate random loads

            px_vecs = (
                torch.rand((n, self.topology.n_free))
                * (self.p_range[0][1] - self.p_range[0][0])
                + self.p_range[0][0]
            )

            py_vecs = (
                torch.rand((n, self.topology.n_free))
                * (self.p_range[1][1] - self.p_range[1][0])
                + self.p_range[1][0]
            )

            pz_vecs = (
                torch.rand((n, self.topology.n_free))
                * (self.p_range[2][1] - self.p_range[2][0])
                + self.p_range[2][0]
            )

            p_mats = torch.stack((px_vecs, py_vecs, pz_vecs), dim=2)

        # print(f"{q_vecs.shape=}")

        # print(f"{p_mats.shape=}")

        X_eq = self.topology.eq_shape_batched(
            q_vecs,
            p_mats,
        )  # n_batches x n_nodes x 3

        # print(f"{X_eq=}")

        # print(f"{self.topology.is_fixed.shape=}")

        # print(f"{self.topology.is_fixed.reshape(-1, 1).expand(n, -1, -1).shape}")

        X_train = torch.cat(
            (
                X_eq,
                self.topology.is_fixed.reshape(-1, 1).expand(n, -1, -1),
                torch.zeros((n, self.topology.n_nodes, 3)),
            ),
            dim=2,
        )

        X_train[:, self.topology.free_ind, 4:] = p_mats

        # print(f"{sample_features=}")

        return X_train, q_vecs


# %%


class ForceDensityPredictorPINN(nn.Module):
    """
    Multiple hidden layer neural network for any-D input/output
    """

    def __init__(
        self,
        topology,
        act_fn=nn.Tanh(),
        hidden_sizes=(10, 10),
        min_q=0.5,
        max_q=30,
    ):
        super(ForceDensityPredictorPINN, self).__init__()

        act_fn = act_fn
        layers = []
        prev_size = topology.n_nodes * 7

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size)

            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

            layers.append(layer)
            layers.append(act_fn)

            prev_size = size

        final_layer = nn.Linear(prev_size, topology.n_edges)
        nn.init.xavier_normal_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        layers.append(final_layer)

        self.net = nn.Sequential(*layers)

        # topology

        self.topology = topology

        # q bounds

        self.register_buffer("min_q", torch.tensor(float(min_q)))
        self.register_buffer("max_q", torch.tensor(float(max_q)))

        # standardization

        self.register_buffer("x_mean", torch.zeros(topology.n_nodes * 7))
        self.register_buffer("x_std", torch.ones(topology.n_nodes * 7))

    def set_input_normalization(self, X):
        flat = X.reshape(X.shape[0], -1)
        self.x_mean = flat.mean(0)
        std = flat.std(0)
        self.x_std = torch.where(
            std < 1e-8, torch.ones_like(std), std
        )  # accounts for constant features

    def forward(self, x):
        # normalize
        x = (x - self.x_mean) / self.x_std

        # Forward pass
        z = self.net(x)
        z = self.min_q + (self.max_q - self.min_q) * torch.sigmoid(z)

        # z = self.min_q + torch.exp(z)
        return z


# %%


def pointwise_Lp_loss(model, Xs_train, _, p=2, trial_q=None):
    """
    Computes a pointwise L^p shape norm between the goal
    nodal locations and the nodal locations caused by the trial
    force densities.

    This loss function can only be used when the goal shape and
    trial topology are exactly the same because it compares
    nodes with the same index.
    """

    n_samples = Xs_train.shape[0]

    target_X = Xs_train[:, :, 0:3]
    p_mats = Xs_train[:, model.topology.free_ind, 4:]

    if trial_q is None:
        trial_q = model(Xs_train.reshape(n_samples, -1))

    hat_X = model.topology.eq_shape_batched(
        trial_q,
        p_mats,
    )

    err = hat_X - target_X

    loss = err.abs().pow(p).sum(dim=(1, 2)).mean()

    err_free = err[:, model.topology.free_ind, :]
    err_free_norm = torch.linalg.norm(err_free, dim=2)

    mean_dist = err_free_norm.mean()
    max_dist = err_free_norm.amax()

    return loss, {"mean_dist": mean_dist.item(), "max_dist": max_dist.item()}


def FDM_physics_loss(
    model,
    _,
    Xs_train,
    trial_q=None,
):
    n_samples = Xs_train.shape[0]
    topo = model.topology

    if trial_q is None:
        trial_q = model(Xs_train.reshape(n_samples, -1))

    X_target = Xs_train[:, :, 0:3]  # n_batches x n_nodes x 3
    p_mats = Xs_train[:, model.topology.free_ind, 4:]  # n_batches x n_free x 3

    edge_vecs = topo.C @ X_target  # n_batches x n_edges x 3
    edge_forces = trial_q.unsqueeze(2) * edge_vecs  # n_batches x n_edges x 3

    residuals = topo.CnT @ edge_forces + p_mats

    loss = residuals.pow(2).mean()

    return loss


# %%

# define some classes for different types of weighting


class LossWeighting:
    """base class with fixed weights"""

    def __init__(self, lambda_data=1.0, lambda_physics=1.0, lambda_bcs=1.0):
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_bcs = lambda_bcs

    def __call__(self, loss_data, loss_physics, loss_bcs, model=None):
        return self.lambda_data, self.lambda_physics, self.lambda_bcs

    def zero_grad(self):
        pass

    def step(self):
        pass


class RelativeWeighting(LossWeighting):
    """Weights higher losses more"""

    def __call__(self, loss_data, loss_physics, loss_bcs, model=None):
        ld, lp, lb = (
            loss_data.detach(),
            loss_physics.detach(),
            loss_bcs.detach(),
        )  # treat weights as constants
        total = ld + lp + lb

        return ld / total, lp / total, lb / total


class SigmoidWeighting(LossWeighting):
    """Self-adaptive weights, w = sigmoid(lambda), lambdas trained by gradient ascent"""

    def __init__(self, lambda_data=1.0, lambda_physics=1.0, lambda_bcs=1.0, lr=1e-3):
        self.lambdas = torch.tensor(
            [lambda_data, lambda_physics, lambda_bcs], requires_grad=True
        )
        self.optimizer = optim.SGD([self.lambdas], lr=lr, maximize=True)

    def __call__(self, loss_data, loss_physics, loss_bcs, model=None):
        w = torch.sigmoid(self.lambdas)
        return w[0], w[1], w[2]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


def train_adaptive(
    model,
    ###
    data_loss_fn,
    data,
    data_generator=None,
    refresh_every=5,  # how often to generate new training data
    data_loss_parameters=(),
    train_val_test=(0.8, 0.1, 0.1),
    batch_size=128,
    batches_per_epoch=20,
    ###
    physics_loss_fn=lambda m, t, X, **kw: torch.zeros(()),
    t_colloc=None,
    physics_loss_parameters=(),
    ###
    bc_loss_fn=lambda m, x, u, X, **kw: torch.zeros(()),
    t_bc=None,
    u_bc=None,
    bc_loss_parameters=(),
    ###
    epochs=500,
    lr=1e-2,
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params={
        "mode": "min",
        "factor": 0.5,
        "patience": 50,
    },
    optimizer=torch.optim.Adam,
    optimizer_params={},
    early_stop_patience=100,
    weighting=None,
    data_start=0,
):
    """
    Train!

    data_generator: if you pass in a data_generator function,
    it's called on each epoch to produce training data.
    The other data will only be used for validation and testing.

    batches_per_epoch: only used if training data is being
    generated on the fly.
    """

    # initialize history
    history = defaultdict(
        list,
        {
            "data_train": [],
            "physics_train": [],
            "bcs_train": [],
            "total_train": [],
            ###
            "data_val": [],
            "physics_val": [],
            "bcs_val": [],
            "total_val": [],
            ###
            "data_test": [],
            "physics_test": [],
            "bcs_test": [],
            "total_test": [],
            ###
            "data_weight": [],
            "physics_weight": [],
            "bcs_weight": [],
            ###
            "lr": [lr],
            ###
            "best_state": None,
            "best_metric": float("inf"),
            "best_epoch": 0,
            ###
            "train_inds": None,
            "val_inds": None,
            "test_inds": None,
        },
    )

    # ---------- weighting scheme ----------
    weighting = weighting or LossWeighting()

    # ---------- choose optimizer ----------
    optimizer = optimizer(model.parameters(), lr=lr, **optimizer_params)

    # scheduler
    scheduler = scheduler(
        optimizer,
        **scheduler_params,
    )

    # ---------- data split ----------
    train_size = int(train_val_test[0] * len(data))
    val_size = int(train_val_test[1] * len(data))
    test_size = len(data) - train_size - val_size

    train, val, test = torch.utils.data.random_split(
        data,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(1),
    )

    train_loader = TensorBatches(train, batch_size=batch_size, shuffle=len(train) > 0)
    val_loader = TensorBatches(val, batch_size=batch_size, shuffle=False)
    test_loader = TensorBatches(test, batch_size=batch_size, shuffle=False)

    history["train_inds"] = train.indices
    history["val_inds"] = val.indices
    history["test_inds"] = test.indices

    # ---------- set normalization scale ----------
    # make sure validation/testing data doesn't bleed into the normalization
    if data_generator is not None:
        X_normalize, _ = data_generator.sample_batch(batch_size * batches_per_epoch)
    else:
        X_normalize = train.dataset.tensors[0][train.indices]

    model.set_input_normalization(X_normalize.reshape(X_normalize.shape[0], -1))

    # ---------- helper for computing losses ----------
    def compute_losses(X, y):
        q = model(X.reshape(X.shape[0], -1))

        # calculate data loss
        loss_data, tracked_metrics = data_loss_fn(
            model,
            X,
            y,
            *data_loss_parameters,
            trial_q=q,
        )

        # calculate physics loss
        loss_physics = physics_loss_fn(
            model,
            t_colloc,
            X,
            *physics_loss_parameters,
            trial_q=q,
        )

        loss_bcs = bc_loss_fn(
            model,
            t_bc,
            u_bc,
            X,
            *bc_loss_parameters,
            trial_q=q,
        )

        return loss_data, loss_physics, loss_bcs, tracked_metrics

    # ---------- train ----------

    epochs_without_improvement = 0

    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        ##### train

        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_total_loss = 0.0

        metrics = defaultdict(int)

        if data_generator is not None and epoch % refresh_every == 0:
            # replace whatever is in train_loader with newly-generated training data
            x_train, y_train = data_generator.sample_batch(
                batch_size * batches_per_epoch
            )
            epoch_train_dataset = TensorDataset(x_train, y_train)
            train_loader = TensorBatches(
                epoch_train_dataset, batch_size=batch_size, shuffle=True
            )

        for X, y in train_loader:
            optimizer.zero_grad()
            weighting.zero_grad()

            # calculate losses
            loss_data, loss_physics, loss_bcs, tracked_metrics = compute_losses(X, y)
            epoch_data_loss += loss_data.item()
            epoch_physics_loss += loss_physics.item()
            epoch_bc_loss += loss_bcs.item()

            # aggregate tracked metrics
            for k, v in tracked_metrics.items():
                metrics[k] += v

            # update weights
            w_data, w_physics, w_bcs = weighting(
                loss_data, loss_physics, loss_bcs, model
            )

            # calculate total loss
            if epoch < data_start:
                total_loss = (
                    (w_data + w_physics + w_bcs)
                    / (w_data + w_bcs)
                    * (w_data * loss_data + w_bcs * loss_bcs)
                )
            else:
                total_loss = (
                    w_data * loss_data + w_physics * loss_physics + w_bcs * loss_bcs
                )

            epoch_total_loss += total_loss.item()

            # backpropagation
            total_loss.backward()
            optimizer.step()
            weighting.step()

        # store weights once per epoch

        history["data_weight"].append(float(w_data))
        history["physics_weight"].append(float(w_physics))
        history["bcs_weight"].append(float(w_bcs))

        # average losses over batches
        epoch_data_loss = epoch_data_loss / len(train_loader)
        epoch_physics_loss = epoch_physics_loss / len(train_loader)
        epoch_bc_loss = epoch_bc_loss / len(train_loader)
        epoch_total_loss = epoch_total_loss / len(train_loader)

        # store losses once per epoch
        history["data_train"].append(float(epoch_data_loss))
        history["physics_train"].append(float(epoch_physics_loss))
        history["bcs_train"].append(float(epoch_bc_loss))
        history["total_train"].append(float(epoch_total_loss))

        # store tracked metrics once per epoch
        for k, v in tracked_metrics.items():
            history[f"{k}_train"].append(metrics[k] / len(train_loader))

        ##### validate

        with torch.no_grad():
            val_loss_data = 0.0
            val_loss_physics = 0.0
            val_loss_bcs = 0.0
            total_val_loss = 0.0

            metrics = defaultdict(int)

            for X_val, y_val in val_loader:
                loss_data, loss_physics, loss_bcs, tracked_metrics = compute_losses(
                    X_val, y_val
                )
                val_loss_data += loss_data.item()
                val_loss_physics += loss_physics.item()
                val_loss_bcs += loss_bcs.item()

                # aggregate tracked metrics
                for k, v in tracked_metrics.items():
                    metrics[k] += v

                # calculate total loss
                if epoch < data_start:
                    total_val_loss += (
                        (w_data + w_physics + w_bcs)
                        / (w_data + w_bcs)
                        * (w_data * loss_data + w_bcs * loss_bcs)
                    )
                else:
                    total_val_loss += (
                        w_data * loss_data + w_physics * loss_physics + w_bcs * loss_bcs
                    )

            # get a metric to compare losses for best model
            # and for patience
            # suspicious if weights are also changing
            # but that's a problem for another day
            metric = total_val_loss

            if epoch >= data_start and metric < history["best_metric"]:
                history["best_metric"] = metric
                history["best_epoch"] = epoch
                history["best_state"] = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0

            elif epoch >= data_start and metric >= history["best_metric"]:
                epochs_without_improvement += 1
                if (
                    epochs_without_improvement >= early_stop_patience
                    and early_stop_patience > 0
                ):
                    print(f"Early stopping at epoch {epoch}")
                    break

            # average validation losses over batches
            val_loss_data = val_loss_data / len(val_loader)
            val_loss_physics = val_loss_physics / len(val_loader)
            val_loss_bcs = val_loss_bcs / len(val_loader)
            total_val_loss = total_val_loss / len(val_loader)

            history["data_val"].append(float(val_loss_data))
            history["physics_val"].append(float(val_loss_physics))
            history["bcs_val"].append(float(val_loss_bcs))
            history["total_val"].append(float(total_val_loss))

            for k, v in tracked_metrics.items():
                history[f"{k}_val"].append(metrics[k] / len(val_loader))

        ##### scheduler
        scheduler.step(total_val_loss)

        # store lr once per epoch
        history["lr"].append(optimizer.param_groups[0]["lr"])

        ##### log
        pbar.set_postfix(
            {
                "Train": f"{epoch_total_loss:0.6f}",
                "Val": f"{total_val_loss:0.6f}",
            }
        )

    # load best model

    model.load_state_dict(history["best_state"])

    # test

    with torch.no_grad():
        test_loss_data = 0.0
        test_loss_physics = 0.0
        test_loss_bcs = 0.0
        total_test_loss = 0.0

        metrics = defaultdict(int)

        for X_test, y_test in test_loader:
            loss_data, loss_physics, loss_bcs, tracked_metrics = compute_losses(
                X_test, y_test
            )
            test_loss_data += loss_data.item()
            test_loss_physics += loss_physics.item()
            test_loss_bcs += loss_bcs.item()

            # aggregate tracked metrics
            for k, v in tracked_metrics.items():
                metrics[k] += v

            # calculate total loss
            total_test_loss += (
                w_data * loss_data + w_physics * loss_physics + w_bcs * loss_bcs
            )

        # average test losses over batches

        test_loss_data = test_loss_data / max(len(test_loader), 1)
        test_loss_physics = test_loss_physics / max(len(test_loader), 1)
        test_loss_bcs = test_loss_bcs / max(len(test_loader), 1)
        total_test_loss = total_test_loss / max(len(test_loader), 1)

        history["data_test"].append(float(test_loss_data))
        history["physics_test"].append(float(test_loss_physics))
        history["bcs_test"].append(float(test_loss_bcs))
        history["total_test"].append(float(total_test_loss))

        for k, v in tracked_metrics.items():
            history[f"{k}_test"].append(metrics[k] / len(test_loader))

    return history


# %%

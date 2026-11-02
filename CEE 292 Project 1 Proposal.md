---
type: hw
course: CEE 292
authors:
  - Wonuola Abiodun
---
I propose a project based off of Rafael Pastrana et. al, *Real-time design of architectural structures with differentiable mechanics and neural networks* ([Arxiv link](https://arxiv.org/abs/2409.02606)). The paper proposes using the force density method (FDM) as a physical basis to approximate goal shell geometries via training a neural network to predict force densities. The force densities can be passed into a linear system solver to determine the exact equilibrium shape, or a PINN can simultaneously solve for the force densities and final geometry. I will re-implement their entire pipeline with the aim to test the model on more advanced structural shapes, e.g. shells with ribs, one-way structural systems, and mixed tension-compression structures. I will generate the dataset myself and use the shape norm and physics loss described in the paper as my benchmarks. I’ll also research nonlinear shell structural models and see if incorporating any of those into the pipeline could produce interesting results.

![[attachments/Pasted-image-20261015095456.png]]

Figure 1 from Pastrana et. al, describing their model architecture.
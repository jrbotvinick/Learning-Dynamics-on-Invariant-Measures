# Learning-Dynamics-on-Invariant-Measures-Using-PDE-Constrained-Optimization

This repository is contains code which can be used to learn an autonomous vector field from trajectory data by solving a PDE-constrained optimization problem [1]. The approach is expected to be particularly helpful when the inference trajectory is sampled slowly and the Lagrangian velocity is difficult to approximate. Here is a summary of the relevant files: 

- `example.ipynb`: Contains an example of our approach where the velocity is paramaterized by global polynomials and the Van der Pol oscillator's vector field is inverted from an observed invariant measure.

- `Comparison`: Contains code for our PDE-constrained optimization-based approach with a neural network paramaterization of the velocity. Also contains code for the SINDy [2] and Neural ODE [3] approaches for learning from trajectory, which can be used for comparison. 

[1] https://arxiv.org/abs/2301.05193 \
[2] https://www.pnas.org/doi/10.1073/pnas.1517384113 \
[3] https://arxiv.org/abs/1806.07366

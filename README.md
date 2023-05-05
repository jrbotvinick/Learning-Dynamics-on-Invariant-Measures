# Learning-Dynamics-on-Invariant-Measures

This repository is contains code which can be used to learn an autonomous vector field from trajectory data by solving a PDE-constrained optimization problem [1]. The approach is expected to be particularly helpful when the inference trajectory is sampled slowly and the Lagrangian velocity is difficult to approximate. Here is a summary of the relevant files: 

- `example.ipynb`: This file contains an example of our approach in which the unkown velocity is paramaterized by global polynomials and the Van der Pol oscillator's vector field is inverted from an observed invariant measure.

- `Comparison`: This folder contains code for our approach with a neural network paramaterization of the velocity. The folder also contains code for the SINDy [2] and Neural ODE [3] approaches for learning dynamics from trajectory data. These files can be used to reproduce the comparisons in [1]. 

The following video demonstrates the process of learning the Van der Pol oscillator's velocity from a polynomial basis using only the invariant measure as inference data. It can be reproduced in the `example.ipynb` file.








https://user-images.githubusercontent.com/100333155/236526309-67c79299-2420-41f5-b98a-ca2c0a41c393.mp4




[1] https://arxiv.org/abs/2301.05193 \
[2] https://www.pnas.org/doi/10.1073/pnas.1517384113 \
[3] https://arxiv.org/abs/1806.07366

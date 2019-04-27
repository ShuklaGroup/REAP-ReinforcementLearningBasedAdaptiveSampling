# REAP: REinforcement learning based AdaPtive sampling
One of the key limitations of Molecular Dynamics (MD) simulations is the computational intractability of sampling protein conformational landscapes associated with either large system size or long time scales. To overcome this bottleneck, we present the REinforcement learning based Adaptive samPling (REAP) algorithm that aims to efficiently sample conformational space by learning the relative importance of each order parameter as it samples the landscape. To achieve this, the algorithm uses concepts from the field of reinforcement learning, a subset of machine learning, which rewards sampling along important degrees of freedom and disregards others that do not facilitate exploration or exploitation. 

In this package, we present a demo of REAP and the original source code used for the proposed algorithm in the publication; "Shamsi, Z., Cheng, K. J., & Shukla, D. (2018). [Reinforcement learning based adaptive sampling: REAPing rewards by exploring protein conformational landscapes.](https://pubs.acs.org/doi/10.1021/acs.jpcb.8b06521) The Journal of Physical Chemistry B".

## Usage
Python code to use REAP algorithm to find the best starting points for each round of simulations.

## Installation instructions
Dependencies:
* numpy
* msmbuilder
* mdtraj

## Tutorial
For a tutorial on how to use REAP on MD simulations and toy potentials to get starting structures for the next simulation round, please see the included file [Tutorial-LShapedLandscape.ipynb](https://github.com/ShuklaGroup/REAP-ReinforcementLearningBasedAdaptiveSampling/blob/master/Tutorial-LShapedLandscape.ipynb).

## Reference
If you use this code, please cite the following paper:

Shamsi, Z., Cheng, K. J., & Shukla, D. (2018). [Reinforcement learning based adaptive sampling: REAPing rewards by exploring protein conformational landscapes.](https://pubs.acs.org/doi/10.1021/acs.jpcb.8b06521) The Journal of Physical Chemistry B, 122(35), 8386-8395.




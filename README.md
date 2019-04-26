# REAP-ReinforcementLearningBasedAdaptiveSampling
One of the key limitations of Molecular Dynamics (MD) simulations is the computational intractability of sampling protein conformational landscapes associated with either large system size or long time scales. To overcome this bottleneck, we present the REinforcement learning based Adaptive samPling (REAP) algorithm that aims to efficiently sample conformational space by learning the relative importance of each order parameter as it samples the landscape. In this package, we present a demo of REAP and the original source code used for the proposed algorithm in the publication; "Shamsi, Z., Cheng, K. J., & Shukla, D. (2018). Reinforcement learning based adaptive sampling: REAPing rewards by exploring protein conformational landscapes. The Journal of Physical Chemistry B.".


## Single round 
```
import RLSim

X_0 = [1,0.5,1.4]
Y_0 = [0.5,0.2,.79]

N = len(X_0)
```
### Run first round of simulation
```
my_sim = RLSim.mockSimulation()
trj1 = my_sim.run([X_0, Y_0])
```
### Choose states which have been discovered whitin the last 5 rounds
#### states with min count or newly discovered
```
trj1_Sp = my_sim.PreSamp(trj1)
```
### Map coordinate space to reaction coorinates space
```
trj1_Sp_theta = my_sim.map(trj1_Sp)
```
### Initialize the weights vector
```
W_0 = [1/2, 1/2]
```
### Update weigths 
```
W_1 = my_sim.updateW(trj1_Sp_theta, W_0)
```
### Get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
Select the starting points for the next round from the least populated states. You need to cluster first and then find clusters with the least population.

```
newPoints_index = my_sim.findStarting(trj1_Sp_theta, W_1, starting_n = N , method = 'RL')
X_1, Y_1 = trj1_Sp[newPoints_index]
trj2 = my_sim.run(X_1, Y_1)
```

### Reference
If you use this code, please cite the following paper:

Shamsi, Z., Cheng, K. J., & Shukla, D. (2018). Reinforcement learning based adaptive sampling: REAPing rewards by exploring protein conformational landscapes. The Journal of Physical Chemistry B, 122(35), 8386-8395.


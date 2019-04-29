## Initialization 
Choose the starting points for the replicas of the first rounds of simulations.

```
import RLSim

X_0 = [1,0.5,1.4]
Y_0 = [0.5,0.2,.79]

N = len(X_0)
```
### Run the first round of simulation
```
my_sim = RLSim.mockSimulation()
trj1 = my_sim.run([X_0, Y_0])
```

### Pick the best starting points for the next round using REAP
Choose states with min count or newly discovered from states discovered whitin the most recent 5 rounds.
 
```
trj1_Sp = my_sim.PreSamp(trj1)
```

Map coordinate space to the reaction coorinates space:
```
trj1_Sp_theta = my_sim.map(trj1_Sp)
```

Initialize the weight vector:
```
W_0 = [1/2, 1/2]
```
Update the weigths:
```
W_1 = my_sim.updateW(trj1_Sp_theta, W_0)
```
Get new starting points (in theta domain) using new reward function based on updated weigths (W_1).
Select the starting points for the next round from the least populated states. You need to cluster first and then find clusters with the least population.

```
newPoints_index = my_sim.findStarting(trj1_Sp_theta, W_1, starting_n = N , method = 'RL')
X_1, Y_1 = trj1_Sp[newPoints_index]
trj2 = my_sim.run(X_1, Y_1)
```

Repeat picking the best starting points using REAP until you find all the landscape.

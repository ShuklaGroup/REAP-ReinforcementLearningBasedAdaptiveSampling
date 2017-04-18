# Reinforcement-Learning-

import RLSim

X_0 = [1,0.5,1.4]
Y_0 = [0.5,0.2,.79]

N = len(X_0)
# run first round of simulation
my_sim = RLSim.mockSimulation()
trj1 = my_sim.run([X_0, Y_0])

# choose states with min count or newly discovered
trj1_Sp = my_sim.PreSamp(trj1)

# map coordinate space to reaction coorinates space
trj1_Sp_theta = my_sim.map(trj1_Sp)

# initialize the weights vector
W_0 = [1/2, 1/2]

# update weigths 
W_1 = my_sim.updateW(trj1_Sp_theta, W_0)

# get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
newPoints_index = my_sim.findStarting(trj1_Sp_theta, W_1, starting_n = N , method = 'RL')

X_1, Y_1 = trj1_Sp[newPoints_index]

trj2 = my_sim.run(X_1, Y_1)


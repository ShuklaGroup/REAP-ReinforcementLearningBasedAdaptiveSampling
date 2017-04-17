import RLSim

X_0 = [1,0.5,1.4]
Y_0 = [1,1.2,1.7]

N = len(X_0)
# run first round of simulation
trj1 = RLSim.mockSimulation.run(X_0, Y_0)

# choose states with min count or newly discovered
trj1_Sp = RLSim.mockSimulation.PreSamp(trj1)

# map coordinate space to reaction coorinates space
trj1_Sp_theta = RLSim.mockSimulation.map(trj1_Sp)

# initialize the weights vector
W_0 = [1/2, 1/2]

# update weigths 
W_1 = RLSim.mockSimulation.updateW(trj1_Sp_theta, W_0)

# get new starting points (in theta domain) using new reward function based on updated weigths (W_1)
newPoints_index = RLSim.mockSimulation.findStarting(trj1_Sp_theta, starting_n = N ,weigths = W_1, method = 'RL')

X_1, Y_1 = trj1_Sp[newPoints_index]

trj2 = RLSim.mockSimulation.run(X_1, Y_1)

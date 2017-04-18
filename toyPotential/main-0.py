import RLSim
import numpy as np 
X_0 = [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01]



N = len(X_0)
# run first round of simulation
my_sim = RLSim.mockSimulation()
trj1 = my_sim.run([X_0, Y_0])
W_0 = [1/2, 1/2]
Ws = []
trjs = trj1
for round in range(25):
	trj1_Sp = my_sim.PreSamp(trj1)
	trj1_Sp_theta = my_sim.map(trj1_Sp)
	
	W_1 = my_sim.updateW(trj1_Sp_theta, W_0)
	W_0 = W_1
	Ws.append(W_0)

	trj1_Sp = my_sim.PreSamp(trjs)
	trj1_Sp_theta = my_sim.map(trj1_Sp)
	
	newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, W_1, starting_n = N , method = 'RL')
	trj2 = my_sim.run(newPoints)
	trj1 = trj2
	trjs = [np.concatenate((trj2[0],trjs[0])), np.concatenate((trj2[1],trjs[1]))]


print(Ws)

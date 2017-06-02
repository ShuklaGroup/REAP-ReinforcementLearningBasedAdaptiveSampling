import RLSim2
import numpy as np 
X_0 = [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01]



N = len(X_0)
# run first round of simulation
my_sim = RLSim2.mockSimulation()
W_0 = [[1/4, 1/4], [1/4, 1/4]]
Ws = []


trj1 = my_sim.run([X_0, Y_0], nstepmax = 10) # run first round
trjs = trj1
trj1_Sp = my_sim.PreSamp(trj1) # least count
trj1_Sp_theta = my_sim.map(trj1_Sp)
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, W_0, starting_n = N , method = 'RL')

trjs_Sp_theta = trj1_Sp_theta
for round in range(40):
	my_sim.updateStat(trjs_Sp_theta)

	W_1 = my_sim.updateW(trj1_Sp_theta, W_0) # rewigth weigths using last round
	#W_1 = my_sim.updateW(trjs_Sp_theta, W_0)
	W_0 = W_1
	Ws.append(W_0)
	print(W_0)
	
	trj1 = my_sim.run(newPoints)
	trj1_Sp = my_sim.PreSamp(trj1)
	trj1_Sp_theta = my_sim.map(trj1_Sp)

	trjs = [np.concatenate((trj1[0],trjs[0])), np.concatenate((trj1[1],trjs[1]))]
	trjs_Sp = my_sim.PreSamp(trjs)
	trjs_Sp_theta = my_sim.map(trjs_Sp)
	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, W_1, starting_n = N , method = 'RL')
	
	
print(Ws)
np.save('w', Ws)

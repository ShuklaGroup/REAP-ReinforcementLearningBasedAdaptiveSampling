import RLSim_leastCount as lc
import numpy as np 
X_0 = [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01]



N = len(X_0) # number of parallel runs 

# run first round of simulation
my_sim = lc.mockSimulation()


# first round
trj1 = my_sim.run([X_0, Y_0], nstepmax = 10)
trj1 = my_sim.PreAll(trj1)

trjs = trj1
trj1_Sp = my_sim.PreSamp(trj1, starting_n = N) # pre analysis
trj1_Sp_theta = my_sim.map(trj1_Sp)
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, starting_n = N , method = 'LC')

trjs_theta = trj1_Sp_theta

trjs_Sp_theta = trj1_Sp_theta
for round in range(40):

	trj1 = my_sim.run(newPoints, nstepmax = 10)
	trj1 = my_sim.PreAll(trj1) # 2 x all points of this round

	com_trjs = []

	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta])))
	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	
	trjs_Sp = my_sim.PreSamp(trjs, starting_n = N)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))
	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, starting_n = N , method = 'LC')
	
	
	

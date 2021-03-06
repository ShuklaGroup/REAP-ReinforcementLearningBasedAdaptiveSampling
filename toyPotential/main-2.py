import RLSim205
import numpy as np 
X_0 = [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01]



N = len(X_0) # number of parallel runs 

# run first round of simulation
my_sim = RLSim205.mockSimulation()
W_0 = [[1/4, 1/4], [1/4, 1/4]] # initial geuss of weights for + - in x and y directions
Ws = [] # series of weights


# first round
trj1 = my_sim.run([X_0, Y_0], nstepmax = 10)
trj1 = my_sim.PreAll(trj1)

trjs = trj1
trj1_Sp = my_sim.PreSamp(trj1, starting_n = N) # pre analysis
trj1_Sp_theta = my_sim.map(trj1_Sp)
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, W_0, starting_n = N , method = 'RL')

trjs_Sp_theta = trj1_Sp_theta
for round in range(10):

	my_sim.updateStat(trjs_Sp_theta) # updates the std and mean

	W_1 = my_sim.updateW(trj1_Sp_theta, W_0) # rewigth weigths using last round
	#W_1 = my_sim.updateW(trjs_Sp_theta, W_0)
	W_0 = W_1
	Ws.append(W_0)
	print('Weight', W_0)
	
	trj1 = my_sim.run(newPoints)
	trj1 = my_sim.PreAll(trj1) # 2 x all points of this round

	trjs = np.array([np.array(np.concatenate((trj1[0],trjs[0]))), np.array(np.concatenate((trj1[1],trjs[1])))])  # 2 x all points
	#print('trjs', trjs)
	#print('trj1', trj1)
	#trjs = my_sim.PreAll(trjs)

	trj1_Sp = my_sim.PreSamp(trjs, starting_n = N)
	trj1_Sp_theta = my_sim.map(trj1_Sp)

    
	trjs_Sp = trj1_Sp
	trjs_Sp_theta = np.array(trj1_Sp_theta)

	print('zz', len(trjs_Sp_theta[0]), len(trjs_Sp[0]))


	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, W_1, starting_n = N , method = 'RL')
	
	
	

print(Ws)
np.save('w', Ws)






import RLSim as rl
import numpy as np 
X_0 = [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01]


X_0 = [1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01]

N = len(X_0) # number of parallel runs 

# run first round of simulation
my_sim = rl.mockSimulation()
W_0 = [[1/4, 1/4], [1/4, 1/4]] # initial geuss of weights for + - in x and y directions
Ws = [] # series of weights


# first round
trj1 = my_sim.run([X_0, Y_0], nstepmax = 10)
trj1 = my_sim.PreAll(trj1)

trjs = trj1
trj1_Sp = my_sim.PreSamp(trj1, starting_n = N) # pre analysis
trj1_Sp_theta = my_sim.map(trj1_Sp)
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, W_0, starting_n = N , method = 'RL')

trjs_theta = trj1_Sp_theta

trjs_Sp_theta = trj1_Sp_theta
for round in range(300):
	# updates the std and mean 
	#my_sim.updateStat(trjs_Sp_theta) # based on min count trajectories
	my_sim.updateStat(trjs_theta) # based on all trajectories
	#W_1 = my_sim.updateW(trj1_Sp_theta, W_0) # rewigth weigths using last round
	W_1 = my_sim.updateW(trjs_Sp_theta, W_0) # important
	W_0 = W_1
	Ws.append(W_0)
	print('Weight', W_0)
	
	trj1 = my_sim.run(newPoints, nstepmax = 10)
	trj1 = my_sim.PreAll(trj1) # 2 x all points of this round

	#trjs = np.array([np.array(np.concatenate((trj1[0],trjs[0]))), np.array(np.concatenate((trj1[1],trjs[1])))])  # 2 x all points
	com_trjs = []

	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta])))
	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	
	trjs_Sp = my_sim.PreSamp(trjs, starting_n = N)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))
  

	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, W_1, starting_n = N , method = 'RL')
	
print(Ws)
np.save('w', Ws)



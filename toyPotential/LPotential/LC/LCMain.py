import LCSim as lc
import numpy as np 

X_0 = [1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01,0.01,0.01,.01]

X_0 = [1.1,1.1,1.1]
Y_0 = [0.01,0.01,.01]

N = len(X_0) # number of parallel runs 
nstepmax = 5
# run first round of simulation
my_sim = lc.mockSimulation()


# first round
trj1 = my_sim.run_noPlt([X_0, Y_0], nstepmax = 10)
trj1 = my_sim.PreAll(trj1)

trjs = trj1
trj1_Sp = my_sim.PreSamp(trj1, starting_n = N, myn_clusters = 20) # pre analysis
trj1_Sp_theta = my_sim.map(trj1_Sp)
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, starting_n = N , method = 'LC')

trjs_theta = trj1_Sp_theta

trjs_Sp_theta = trj1_Sp_theta

count = 1
for round in range(700):

	trj1 = my_sim.run_noPlt(newPoints, nstepmax = nstepmax)
	trj1 = my_sim.PreAll(trj1) # 2 x all points of this round

	com_trjs = []

	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta])))
	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	
	trjs_Sp = my_sim.PreSamp(trjs, starting_n = N)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))
	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, starting_n = N , method = 'LC')
	count = count + 1 

	print(np.any(trjs_theta[1]>1.5))
	if np.any(trjs_theta[1]>1.4):
		import matplotlib.pyplot as plt
		plt.plot(trjs_theta[0], trjs_theta[1], 'o')
		plt.savefig('fig.png')
		np.save('count', count)
		break

#import matplotlib.pyplot as plt
#plt.plot(trjs_theta[0], trjs_theta[1], 'o')
#plt.savefig('fig_all.png')
my_sim.pltFinalPoints(trjs_theta)
np.save('Trjs', trjs_theta)	
np.save('count_all', count)	
	
	

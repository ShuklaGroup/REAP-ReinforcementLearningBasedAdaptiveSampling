import SLSim as sl
import numpy as np 


X_0 = [0.3, 0.3, 0.3, 0.3, 0.3]
Y_0 = [0.6, 0.6, 0.6, 0.6, 0.6]

N = len(X_0) # number of parallel runs 

nstepmax = 20

# run first round of simulation
my_sim = sl.mockSimulation()


# first round
# first round
init = [X_0, Y_0]
print(len(init), len(init[0]))
trj1 = my_sim.run_noPlt(init, nstepmax = nstepmax)
init = [trj1[0][-1][0], trj1[1][-1][0]]
trj1 = my_sim.PreAll(trj1)
trjs = trj1

for i in range(51):
	my_sim.pltPoints(trjs, init, i)
	trj1 = my_sim.run_noPlt(init, nstepmax = nstepmax)
	init = [trj1[0][-1][0], trj1[1][-1][0]]
	print(len(init), len(init[0]))
	trj1 = my_sim.PreAll(trj1)
	com_trjs = []

	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta])))
	trjs = np.array(com_trjs)
	
np.save('Trjs', trjs)	

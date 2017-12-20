import LCSim as lc
import numpy as np 

X_0 = [1.1]
Y_0 = [0.01]

N = len(X_0) # number of parallel runs 
nstepmax = 15*26*24
nstepmax = 3*5*700+3*10
# run first round of simulation
my_sim = lc.mockSimulation()


# first round
trj1 = my_sim.run_noPlt([X_0, Y_0], nstepmax = nstepmax)
trj1 = my_sim.PreAll(trj1)

trjs = trj1
count = 1
my_sim.pltFinalPoints(trj1)
np.save('Trjs', trj1)	
np.save('count_all', count)	

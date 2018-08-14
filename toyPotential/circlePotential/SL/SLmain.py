import SLSim as lc
import numpy as np 

X_0 = [0.01]
Y_0 = [0.01]

#X_0 = [1.1]
#Y_0 = [0.01]

N = len(X_0) # number of parallel runs 
nstepmax = 15*26*24
nstepmax = 10*5*300+10*10
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

#import matplotlib.pyplot as plt
#plt.plot(trjs_theta[0], trjs_theta[1], 'o')
#plt.savefig('fig_all.png')

#my_sim.pltFinalPoints(trjs_theta)
#np.save('Trjs', trjs_theta)	
#np.save('count_all', count)	
	
	

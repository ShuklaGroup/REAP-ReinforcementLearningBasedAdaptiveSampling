# python 2.7 MSMbuilder 2.8

from scipy import io as sio
sio.mmwrite('transmat.mtx', msm.transmat_)



import LCSim as lc
import numpy as np 

S_0 = [0]

N = len(X_0) # number of parallel runs 
nstepmax = 15*26*24
nstepmax = 3*5*700+3*10

# run simulation
my_sim = lc.mockSimulation()
trj1 = my_sim.run(S_0, nstepmax = nstepmax)
trj1 = my_sim.PreAll(trj1)

trjs = trj1

my_sim.pltPoints(trj1)
np.save('Trjs', trj1)	

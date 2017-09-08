# python 2.7 MSMbuilder 2.8

from scipy import io as sio
import SLSim as sl
import numpy as np 

# inital state
S_0 = [0] # inactive

#S_0 = [1728] # active
N = len(S_0) # number of parallel runs 
nstepmax = 20000
print(nstepmax*0.005)
# run simulation
my_sim = sl.mockSimulation()
my_sim.tp = sio.mmread('tProb.mtx')
my_sim.x = np.load('Gens_aloopRMSD.npy')
my_sim.y = np.load('Gens_y_deltaDist.npy')
my_sim.mapping = np.load('map_Macro2micro.npy')

trj1 = my_sim.run(S_0, nstepmax = nstepmax)
trj1 = my_sim.PreAll(trj1)

trjs = trj1[0]


trj_x, trj_y = my_sim.map(trjs)
my_sim.pltPoints(trj_x, trj_y)

np.save('Trjs', trj1)	



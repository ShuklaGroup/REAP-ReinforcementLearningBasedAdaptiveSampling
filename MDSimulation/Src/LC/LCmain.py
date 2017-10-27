# python 2.7 MSMbuilder 2.8

from scipy import io as sio
import LCSim as lc
import numpy as np 

# inital state
S_0 = [0] # inactive

N = len(S_0) # number of parallel runs 
nstepmax = 80
print('Simulation length: ', nstepmax*0.005)

# Simulation setup
my_sim = lc.mockSimulation()
my_sim.tp = sio.mmread('tProb.mtx')
my_sim.x = np.load('Gens_aloopRMSD.npy')
my_sim.y = np.load('Gens_y_KE.npy')
my_sim.mapping = np.load('map_Macro2micro.npy')

#### first round
trj1 = my_sim.run(S_0, nstepmax = nstepmax) # 1 x 1 x n_frames
trj1 = my_sim.PreAll(trj1) # 1 x 1 x n_frames
trjs = trj1[0] # 1 x n_frames
trj1_Sp = my_sim.PreSamp_MC(trjs, N = 1) # pre analysis # 1 x n_samples
trj1_Sp_theta = np.array(my_sim.map(trj1_Sp)) # [trjx, trjy]
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, starting_n = N , method = 'LC') # needs 2 x something ###### CHANGE
trjs_theta = trj1_Sp_theta
trjs_Sp_theta = trj1_Sp_theta


for round in range(500):
	oldTrjs = trjs
	trj1 = my_sim.run(newPoints, nstepmax = 20)
	trj1 = my_sim.PreAll(trj1)[0] # 2 x all points of this round
	com_trjs = np.concatenate((trjs, trj1))
	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	
	trjs_Sp = my_sim.PreSamp_MC(trjs, N = 1)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))
	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, starting_n = N , method = 'LC') #### CHANGE !!!!!


my_sim.pltPoints(trjs_theta[0], trjs_theta[1])
np.save('trjs_theta_LC', trjs_theta)

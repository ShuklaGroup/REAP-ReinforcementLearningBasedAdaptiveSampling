# python 2.7 MSMbuilder 2.8

from scipy import io as sio
import RLSim as rl
import numpy as np 

# inital state
S_0 = [0] # inactive
n_op = 4 # number of order parameters

N = len(S_0) # number of parallel runs 
nstepmax = 80
print('Simulation length: ', nstepmax*0.005)
# W_0 = [[1/4, 1/4], [1/4, 1/4]] # initial geuss of weights for + - in x and y directions
# W_0 = [0.5, 0.5] 
W_0 = 1/n_op*np.ones(n_op)

Ws = [] # time series of weights
Ws.append(W_0)
print('Weight', W_0)
# run simulation
my_sim = rl.mockSimulation()
my_sim.tp = sio.mmread('tProb.mtx')
my_sim.x1 = np.load('Gens_aloopRMSD.npy')
my_sim.x2 = np.load('Gens_y_deltaDist.npy')
my_sim.x3 = np.load('Gens_x3.npy')
my_sim.x4 = np.load('Gens_x4.npy')

my_sim.mapping = np.load('map_Macro2micro.npy')

#### first round
trj1 = my_sim.run(S_0, nstepmax = nstepmax) # 1 x 1 x n_frames
trj1 = my_sim.PreAll(trj1) # 1 x 1 x n_frames
trjs = trj1[0] # 1 x n_frames
trj1_Sp = my_sim.PreSamp_MC(trjs, N = 20) # pre analysis # 1 x n_samples

trj1_Sp_theta = np.array(my_sim.map(trj1_Sp)) # [trjx1, trjx2, ...] 
newPoints = my_sim.findStarting(trj1_Sp_theta, trj1_Sp, W_0, starting_n = N , method = 'RL') # needs 2 x something   ### Should work fine
trjs_theta = trj1_Sp_theta
trjs_Sp_theta = trj1_Sp_theta

for round in range(500):
	# updates the std and mean 
	my_sim.updateStat(trjs_theta) # based on all trajectories  ### Should work fine

	W_1 = my_sim.updateW(trjs_Sp_theta, W_0) # important ## Should change with differet OP
	W_0 = W_1
	Ws.append(W_0)
	print('Weight', W_0)
	
	oldTrjs = trjs
	trj1 = my_sim.run(newPoints, nstepmax = 20)
	trj1 = my_sim.PreAll(trj1)[0] # 2 x all points of this round
	com_trjs = np.concatenate((trjs, trj1))
	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	
	trjs_Sp = my_sim.PreSamp_MC(trjs, N = 20)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))
	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, W_1, starting_n = N , method = 'RL')

my_sim.pltPoints(trjs_theta[0], trjs_theta[1])
print(Ws)
np.save('w_2', Ws)
np.save('trjs_theta_2', trjs_theta)




#################

## Should change with differet OP 
# map
# updateW


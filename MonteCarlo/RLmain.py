# B2AR
import RLSim as rl
import numpy as np 

msm = pickle.load(open('MSM.pkl','rb'))

init = 132
N = 10 # number of parallel runs 
inits = [init for i in range(N)]


# run first round of simulation
my_sim = rl.mockSimulation()
my_sim.msm = msm
#W_0 = [[1/4, 1/4], [1/4, 1/4]] # initial geuss of weights for + - in x and y directions
n_ec = 800
W_0 = [1/n_ec for i in range(n_ec)]
Ws = [] # series of weights


# first round
trj1 = my_sim.run(inits, nstepmax = 10)
comb_trj1 = np.concatenate(trj1)
#trj1 = my_sim.PreAll(trj1)

# comb_trj1 = first trajectory in the format of 1x(N*step) and cluster labels
trjs = comb_trj1

# trj1_Ps= presampled (least count,...) from first trajectory in the format of cluster labels
trj1_Ps = my_sim.PreSamp_MC(trj1, N = 3*N) # pre analysis

# trj1_Ps_theta = presampled (least count,..) from first trajectory in the format of n_ec x n_frames and theta (evolutionary couplings,..)
trj1_Ps_theta = my_sim.map(trj1_Ps)

# 
newPoints = my_sim.findStarting(trj1_Ps_theta, trj1_Ps, W_0, starting_n = N , method = 'RL')

trjs_theta = trj1_Sp_theta

trjs_Sp_theta = trj1_Sp_theta

count = 1 
for round in range(150):
	# updates the std and mean 
	my_sim.updateStat(trjs_theta) # based on all trajectories
	W_1 = my_sim.updateW(trjs_Sp_theta, W_0)
	W_0 = W_1
	Ws.append(W_0)
	print('Weight', W_0)
	
	trj1 = my_sim.run_noPlt(newPoints, nstepmax = 10)
	trj1 = my_sim.PreAll(trj1) # 2 x all points of this round

	com_trjs = []

	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta])))

	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	
	trjs_Sp = my_sim.PreSamp(trjs, starting_n = N)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))
	
	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, W_1, starting_n = N , method = 'RL')

	count = count + 1 

	print(np.any(trjs_theta[1]>1.4))
	if np.any(trjs_theta[1]>1.4):
		import matplotlib.pyplot as plt
		plt.plot(trjs_theta[0], trjs_theta[1], 'o')
		plt.savefig('fig.png')
		np.save('count', count)
		break
	
	
import matplotlib.pyplot as plt
plt.plot(trjs_theta[0], trjs_theta[1], 'o')
plt.savefig('fig_all.png')
np.save('count_all', count)


print(Ws)
np.save('w', Ws)


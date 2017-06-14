# B2AR
import pickle
import RLSim as rl
import numpy as np

msm =  pickle.load(open('MSM150.pkl','rb'))
init = 132
N = 10 # number of parallel runs 
inits = [init for i in range(N)]
my_sim = rl.mockSimulation()
my_sim.msm = msm
n_ec = 10
W_0 = [1/n_ec for i in range(n_ec)]

Ws = [] # series of weights


# first round
trj1 = my_sim.run(inits, nstepmax = 10)
comb_trj1 = np.concatenate(trj1)
#trj1 = my_sim.PreAll(trj1)

# comb_trj1 = first trajectory in the format of 1x(N*step) and cluster labels
trjs = comb_trj1

# trj1_Ps= presampled (least count,...) from first trajectory in the format of cluster labels
trj1_Ps = my_sim.PreSamp_MC(trj1, N = 3*N) # pre analysis , 1 x n_frames

# trj1_Ps_theta = presampled (least count,..) from first trajectory in the format of n_ec x n_frames and theta (evolutionary couplings,..)
trj1_Ps_theta = my_sim.map(trj1_Ps)
# 
newPoints = my_sim.findStarting(trj1_Ps_theta, trj1_Ps, W_0, starting_n = N , method = 'RL')
trjs_theta = trj1_Ps_theta
trjs_Ps_theta = trj1_Ps_theta
count = 1 

for round in range(20):
        # updates the std and mean 
        my_sim.updateStat(trjs_theta) # based on all trajectories
        W_1 = my_sim.updateW(trjs_Ps_theta, W_0)
        W_0 = W_1
        Ws.append(W_0)
        print('Weight', W_0)
        
        trj1 = my_sim.run(newPoints, nstepmax = 10) # N (number of parallel) x n_all_frames
        trj1 = np.concatenate(trj1) # 1 x n_all_frames
	
	
        
        # combine trj1 with old trajectories
        com_trjs = np.concatenate((trjs, trj1))
        
        trjs = np.array(com_trjs)
        trjs_theta = np.array(my_sim.map(trjs))
        
        trjs_Ps = my_sim.PreSamp_MC(trjs, N = 4*N)
        trjs_Ps_theta = np.array(my_sim.map(trjs_Ps))
        
        newPoints = my_sim.findStarting(trjs_Ps_theta, trjs_Ps, W_1, starting_n = N , method = 'RL')

        count = count + 1 

#       print(np.any(trjs_theta[1]>1.4))
#       if np.any(trjs_theta[1]>1.4):
#               import matplotlib.pyplot as plt
#               plt.plot(trjs_theta[0], trjs_theta[1], 'o')
#               plt.savefig('fig.png')
#               np.save('count', count)
#               break
        
        
import matplotlib.pyplot as plt
plt.plot(trjs_theta[0], trjs_theta[1], 'o')
plt.savefig('fig_all.png')
np.save('count_all', count)


print(Ws)
np.save('w', Ws)

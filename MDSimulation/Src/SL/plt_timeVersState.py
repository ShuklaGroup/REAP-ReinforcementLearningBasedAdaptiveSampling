import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
    
data_rl = np.array(np.load('RL/n_discoveredS_time.npy'))   
data_sl = np.array(np.load('SL/n_discoveredS_time.npy'))


plt.plot(0.005*data_sl[:,0], data_sl[:, 1], lw=1.5, color="orangered", label='Single long trajectory')
plt.fill_between(0.005*data_sl[:,0], data_sl[:, 1], color="orangered", linewidth=0.0, alpha=0.4)

plt.plot(0.005*data_rl[:,0], data_rl[:, 1], lw=1.5, color="midnightblue", label='REAP trajectories')
plt.fill_between(0.005*data_rl[:,0], data_rl[:, 1], color="midnightblue", linewidth=0.0, alpha=0.4)

plt.legend(loc=0, fontsize=18)
plt.xlim([0, 15])
plt.xticks([0, 5, 10, 15])
plt.yticks([0, 400, 800])

plt.ylabel('Number of discovered states')
plt.xlabel('Time ('+ r'$\mu$'+'s)')

plt.savefig('n_discoveredS_time.png')
plt.show()



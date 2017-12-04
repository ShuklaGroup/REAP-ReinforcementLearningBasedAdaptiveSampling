import numpy as np

xedges = np.linspace(-180, 180, num=181)
yedges = np.linspace(-180, 180, num=181)

phi_all = np.load('phi.npy')
psi_all = np.load('psi.npy')

phi = []
psi = []

data = []

for frame in range(len(phi_all)):
    phi.append(phi_all[frame])
    psi.append(psi_all[frame])
    H, xedges, yedges = np.histogram2d(phi, psi, bins=(xedges, yedges))
    H0 = np.unique(np.concatenate(H))
    n_states = len(H0)-1
    data.append([frame, n_states])

np.save('n_discoveredS_time.npy', data)

#####
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


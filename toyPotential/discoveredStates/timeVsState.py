#######
import numpy as np

for i in range(100):
	xedges = np.linspace(-0.5, 1.5, num=51)
	yedges = np.linspace(-0.5, 1.5, num=51)
	stride = 1 # 
	w = np.load('trjs_theta'+str(i)+'.npy')[::stride]
	phi_all = w[0]
	psi_all = w[1]

	phi = []
	psi = []

	data = []

	for frame in range(len(phi_all)):
	    phi.append(phi_all[frame])
	    psi.append(psi_all[frame])
	    H, xedges, yedges = np.histogram2d(phi, psi, bins=(xedges, yedges))
	    H0 = np.nonzero(np.concatenate(H))
	    n_states = len(H0[0])
	    data.append([frame, n_states])
	np.save('n_discoveredS_time'+str(i)+'.npy', data)




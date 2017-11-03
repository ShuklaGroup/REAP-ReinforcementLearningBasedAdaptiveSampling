#######
import numpy as np
import SLSim as sl
from scipy import io as sio

stride = 1 # 
my_sim = sl.mockSimulation()
my_sim.tp = sio.mmread('tProb.mtx')
my_sim.x = np.load('Gens_aloopRMSD.npy')
my_sim.y = np.load('Gens_y_KE.npy')
my_sim.mapping = np.load('map_Macro2micro.npy')

for i in range(100):
	try:
		xedges = np.linspace(0, 1, num=41)
		yedges = np.linspace(0, 2, num=41)

		trjs = np.load('trjs_theta'+str(i)+'.npy')
		#trj_x, trj_y = my_sim.map(trjs)
	
		phi_all = trjs[0]
		psi_all = trjs[1]
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
	except:
		print(i)
		




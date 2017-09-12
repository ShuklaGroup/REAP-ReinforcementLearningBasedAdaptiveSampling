# initialize
import glob
import numpy as np
import mdtraj as md

topFile = 'csrc_inactive.pdb'
top = md.load(topFile).topology

# select the atoms
a1 = top.select('name CD and resid 50 and resname GLU')
b1 = top.select('name NZ and resid 35 and resname LYS')

a2 = a1
b2 = top.select('name CZ and resid 149 and resname ARG')
pairs = [[a1[0],b1[0]],[a2[0],b2[0]]]


# calculate
for file in glob.glob('*/*.lh5'):
	print(file)
	t = md.load(file)
	# distances
	dist1 = md.compute_distances(t,pairs)
	np.save(file.replace('.lh5','_KE-RE.npy'), dist1)
    
    
#### for Gens
re = [delta[i][1] for i in range(2000)]
ke = [delta[i][0] for i in range(2000)]
np.save('Gens_KE.npy', ke)
np.save('Gens_RE.npy', re)

#### for Monte Carlo trajectory
import glob
import numpy as np
import mdtraj as md

file = 'MSM_traj_csrc_100microsecs.pdb'
trj = md.load(file)
top = trj.topology

a1 = top.select('name CD and resid 50 and resname GLU')
b1 = top.select('name NZ and resid 35 and resname LYS')

a2 = a1
b2 = top.select('name CZ and resid 149 and resname ARG')

pairs = [[a1[0],b1[0]],[a2[0],b2[0]]]
dist2 = md.compute_distances(t,pairs)
re = [dist2[i][1] for i in range(trj.n_frames)]
ke = [dist2[i][0] for i in range(trj.n_frames)]

np.save('MSM_traj_csrc_100microsecs_KE.npy', ke)
np.save('MSM_traj_csrc_100microsecs_RE.npy', re)




import glob
import numpy as np
import mdtraj as md

topFile = '../csrc_inactive.pdb'
trj = md.load(topFile)
top = trj.topology

a1 = top.select('name CA')
b1 = top.select('name CA')

pairs = top.select_pairs(selection1=a1, selection2=b1)

for file in glob.glob('Gens.lh5'):
	print(file)
	t = md.load(file)
	dist2 = md.compute_distances(t,pairs)
	print(len(dist2), trj.n_frames)
	np.save(file.replace('.lh5', '_CA-distances.npy'), dist2)


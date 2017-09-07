import glob
import numpy as np
import mdtraj as md

topFile = 'csrc_inactive.pdb'
top = md.load(topFile).topology

# aloop
a1 = top.select('name CD and resid 50 and resname GLU')
b1 = top.select('name NZ and resid 35 and resname LYS')

a2 = a1
b2 = top.select('name CZ and resid 149 and resname ARG')
pairs = [[a1[0],b1[0]],[a2[0],b2[0]]]


#############
for file in glob.glob('*/*.lh5'):
	print(file)
	t = md.load(file)
	# distances
	dist1 = md.compute_distances(t,pairs)
	
	np.save(file.replace('.lh5','_KE-RE.npy'), dist1)
    
    

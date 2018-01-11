import glob
import numpy as np
import mdtraj as md

topFile = '../csrc_inactive.pdb'
trj = md.load(topFile)
top = trj.topology

a1 = top.select('name CA')
b1 = top.select('name CA')
pairs = top.select_pairs(selection1=a1, selection2=b1)

dist1_1 = top.select('name CA and resid 109 and resname GLN')
dist1_2 = top.select('name CA and resid 112 and resname SER')
pairs_i = np.where((pairs[:,0]==dist1_1) & (pairs[:,1]==dist1_2))

ca = np.load('Gens_CA-distances.npy')
ca_r1=ca[:, pairs_i]
np.save('Gens_clob-1', ca_r1)

# ?!!!!!
import numpy as np 
import os
n_states = 999
n_samples = 50 
n_ec = 50
try:
    os.mkdir('MSMStatesDVals_'+str(n_ec))
except:
    print('directory exists')

ref = np.loadtxt('ref_dist')
dist_ref = ref[:n_ec]
for state in range(n_states):
    filename = 'MSMStatesDVals_'+str(n_ec)+'/cluster'+str(state)
    dists=[]
    for j in range(n_samples):
        dist = np.loadtxt('MSMStatesAllVals_-1/cluster'+str(state)+'-'+str(j))
        dist = dist[:n_ec]
        dists.append(dist)
        
dist_ave = np.average(dists, axis=0)
dist_dif = np.absolute(dist_ave-dist_ref)
val = np.sum(dist_dif)
np.savetxt(filename, [val])
print(' saved!'+str(val))

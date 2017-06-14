
def readCouplings(couplingFN, n_EC=5000, revFlag=False, scores=False):
	"""
	Parameters
	----------
	couplingFN: 
		the name of the file containing coupling information
	n_EC:
		number of evolutionary coupling
	revFlag:
		it'll False if top couplings are using, and if True, the least ones are using (reverse)
	scores:
		If it's True, it'll return the scores beside the couples and if it's False, it wont return scores 
	output :
		an array containing coupling residues
	"""
	import numpy as np
	cp = np.loadtxt(couplingFN, delimiter=',')
	if scores:
		coupling = cp[:,0:3]
	else:
		coupling = cp[:,0:2]
	if revFlag:
		l = len(coupling)
		start = l - n_EC
		coupling = coupling[start:-1,:]
	else:
		coupling = coupling[0:n_EC,:]
	
	return coupling
        

 def refineCouplingIndx(trj, coupling, scFlag=False):
	"""
	Parameters
	----------
	trj : 
		trajectory or strucure in mdtraj format which is the reference for refining the couplings
	coupling :
		an array of un refined couplings
	scFlag :
		if True, the coupling has scoring information and is 3D if False its 2D
	return :
		refined coupling array
	"""
	coupleIndx = []
	for couples in coupling:
		a = resIndex(trj,couples[0])
		b = resIndex(trj,couples[1])
		if a==[] or b==[]:
			print("ERROR IN CONVERTING COUPLING")
		else:
			if scFlag:
				coupleIndx.append([a, b, couples[2]])
			else:
				coupleIndx.append([a, b])
	return coupleIndx   

import mdtraj as md
import numpy as np

referenceStructure='ww_index_ref.pdb'
couplingFN = 'Fip35_CouplingScores.csv'
coupling = readCouplings(couplingFN)

ref = md.load(referenceStructure)
couplingIndxRef = refineCouplingIndx(ref, coupling)

for cl in range(n_clusters):
        dists = []
        for s in range(n_samples):
                pdb_name = '/Users/ZahraSh/Desktop/Projects/MSM\ Adaptive\ Sampling/Folding-ww/revision1/sample_pdbs/cluster'+str(cl)+'_'+str(s)+'.pdb '
                str = md.load(pdb_name)
                dist = md.compute_contacts(str, couplingIndxRef, scheme='closest-heavy')
                dists.append(dist)
        dists = np.array(dists)
        ave_dist = np.average(dists, axis=0)
        np.savetxt('Ave_AllECdist_cluster'+str(cl), ave_dist)
                
                        
       
        
    

from msmbuilder.decomposition import tICA
import numpy as np
import pickle
from msmbuilder.utils import io 
import msmbuilder.cluster
import glob 
from msmbuilder.msm import MarkovStateModel


# setting the parameters
n_components= 8
lag_time= 10 
n_clusters= 2000
sys = 'Src-'
n_timescales = 10
lagTime = 50 # 5ns

# loading the data
dataset = []
import glob
for file in glob.glob('highRMSF_phi_psi/*.npy'):
     a = np.array(np.load(file))
     dataset.append(a)

# building tica
tica = tICA(n_components=n_components , lag_time=lag_time)
tica.fit(dataset)
tica_traj = tica.transform(dataset)
pickle.dump(tica, open(sys+'_tICs_'+str(n_components)+'.pkl','wb'))

# clustering
states = msmbuilder.cluster.KMeans(n_clusters = n_clusters)
states.fit(tica_traj)
io.dump(states, sys+'_tICs_'+str(n_components)+'nCluster_'+str(n_clusters)+'.pkl')

# making MSM
msm = MarkovStateModel(lag_time=lagTime, n_timescales=n_timescales)
msm.fit_transform(cl.labels_)
io.dump(msm, 'MSM'+sys)

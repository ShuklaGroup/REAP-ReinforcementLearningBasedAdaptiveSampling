import numpy as np

z = np.load('Trjs.npy')[0]

states = []
data = []

for i in range(len(z)):
    states.append(z[i])
    states = np.unique(states)
    n_discoveredS = len(states)
    time = i
    data.append([time, n_discoveredS])
    
np.save('n_discoveredS_time.npy', data)    


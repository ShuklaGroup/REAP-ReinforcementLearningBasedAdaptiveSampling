import matplotlib.pyplot as plt
import numpy as np

w = np.load('w.npy')
plt.plot(np.arange(len(w[:,0,0])), w[:,0,0])
plt.plot(np.arange(len(w[:,0,1])), w[:,0,1])
plt.plot(np.arange(len(w[:,1,0])), w[:,1,0])
plt.plot(np.arange(len(w[:,1,1])), w[:,1,1])
plt.show()

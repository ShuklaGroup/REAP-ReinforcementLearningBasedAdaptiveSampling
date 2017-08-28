
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)


w = np.load('w.npy')

y = w[:,1,0] + w[:,1,1]
x = w[:,0,0] + w[:,0,1]


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

ax.plot(np.arange(len(w[:,0,0])), x, 'r', lw=2, label='X weight')
ax.plot(np.arange(len(w[:,0,1])), y, 'black', lw=2, label='Y weight')
ax.set_ylim([0, 1])
ax.set_xlabel('Frame number')
ax.set_ylabel('Weight')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('test.png')
plt.show()




"""
import matplotlib.pyplot as plt
import numpy as np
w = np.load('w.npy')
plt.plot(np.arange(len(w[:,0,0])), w[:,0,0])
plt.plot(np.arange(len(w[:,0,1])), w[:,0,1])
plt.plot(np.arange(len(w[:,1,0])), w[:,1,0])
plt.plot(np.arange(len(w[:,1,1])), w[:,1,1])
plt.show()
"""


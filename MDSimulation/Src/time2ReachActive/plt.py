import matplotlib.pyplot as plt
import numpy as np


LC = np.load('LC/LC-activationTimes.npy')
LC = 0.005*LC

RL = np.load('RL/RL-activationTimes.npy')
RL = 0.005*RL

SL = np.load('SL/SL-activationTimes.npy')
SL = 0.005*SL

fig, axes = plt.subplots(nrows=1, ncols=1)
axes.boxplot([LC, RL])
plt.savefig('fig')
plt.show()


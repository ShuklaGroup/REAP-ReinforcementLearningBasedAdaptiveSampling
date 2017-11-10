#import matplotlib.pyplot as plt
import numpy as np
import pylab as plt
import seaborn as sns


plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)


LC = np.load('LC/LC-activationTimes.npy')
LC = 0.005*LC

RL = np.load('RL/RL-activationTimes.npy')
RL = 0.005*RL

SL = np.load('SL/SL-activationTimes.npy')
SL = 0.005*SL

sns.set_style("whitegrid")


ax1= sns.boxplot(data=[RL, LC, SL], palette='spectral',width=.28)
#ax1 = sns.swarmplot(data=[LC,RL], size=6, edgecolor="black", linewidth=.9)

#plt.xticks(rotation=-20)
ax1.set(xticklabels=['REAP', 'Least Count', 'Single Long'])

#plt.xlabel('Portion of landscape discovered')
ax1.set_ylabel('Time to reach to active ('+ r'$\mu$'+'s)', fontdict={'fontsize' : 20})
#ax1.set_yticks([0, 20, 40])

plt.savefig('fig4')
plt.show()


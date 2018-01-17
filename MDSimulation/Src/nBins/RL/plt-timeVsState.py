import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
totalData = 50*60

RLmeans = np.load('RL/means.npy')
RLstds = np.load('RL/stds.npy')

LCmeans = np.load('LC/means.npy')
LCstds = np.load('LC/stds.npy')

SLmeans = np.load('SL/means.npy')
SLstds = np.load('SL/stds.npy')

xdata = range(totalData)
xdata = 0.005*np.array(xdata)
RLydata = RLmeans
RLyerror = RLstds
LCydata = LCmeans
LCyerror = LCstds
SLydata = SLmeans
SLyerror = SLstds

color="midnightblue"
plt.plot(xdata, RLydata, color=color, lw=2.5, label='REAP trajectories')
#plt.fill_between(xdata, RLydata-RLyerror, RLydata+RLyerror, alpha=0.4, color=color)


color="green"
plt.plot(xdata, LCydata, color=color, lw=2.5, label='Least Count based trajectories')
#plt.fill_between(xdata, LCydata-LCyerror, LCydata+LCyerror, alpha=0.4, color=color)

color="orangered"
plt.plot(xdata, SLydata, color=color, lw=2.5, label='Single long trajectories')
#plt.fill_between(xdata, SLydata-SLyerror, SLydata+SLyerror, alpha=0.4, color=color)

plt.legend(loc=4, fontsize=20, frameon=False)

plt.xlim([0, 15])
plt.xticks([0,5, 10, 15])

plt.yticks([0, 95, 190], ['0', '0.5', '1'])
plt.ylim([0,190])
plt.ylabel('Portion of landscape discovered')
plt.xlabel('Time ('+ r'$\mu$'+'s)')
plt.tight_layout()
plt.savefig('n_discoveredS_REAP2.png', dpi = 300)
plt.show()

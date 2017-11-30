import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

RLmeans = np.load('means.npy')
RLstds = np.load('stds.npy')



xdata = range(len(RLmeans))
xdata = 0.005*np.array(xdata)
RLydata = RLmeans
RLyerror = RLstds


color="midnightblue"
plt.plot(xdata, RLydata, color=color, lw=1.5, label='REAP trajectories')
plt.fill_between(xdata, RLydata-RLyerror, RLydata+RLyerror, alpha=0.4, color=color)

"""
color="orangered"
plt.plot(xdata, SLydata, color=color, lw=1.5, label='Single long trajectory')
plt.fill_between(xdata, SLydata-SLyerror, SLydata+SLyerror, alpha=0.4, color=color)
"""
plt.legend(loc=2, fontsize=20, frameon=False)
#plt.xlim([0, 0.005*3000])
#plt.ylim([0, 800])

#plt.xlim([0, 15])
#plt.xticks([0, 5, 10, 15])
#plt.yticks([0, 200, 400, 600, 800])

plt.ylabel('Number of discovered states')
plt.xlabel('Time ('+ r'$\mu$'+'s)')

plt.savefig('n_discoveredS_time.png', dpi = 300)
plt.show()

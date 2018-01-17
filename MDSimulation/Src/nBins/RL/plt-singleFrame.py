import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


plt.rcParams.update({'font.size':20})
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
p1=np.load('LC/fulldata.npy')

p1last = p1[2999]

p2=np.load('RL/fulldata.npy')
p2last = p2[2999]

p3=np.load('SL/fulldata.npy')
p3last = p3[2999]

print(p2last)

plt.hist(p1last, 50,  facecolor='green', alpha=0.5, label='Least Count based', edgecolor = "none")
plt.hist(p2last, 50,  facecolor="midnightblue", alpha=0.5, label='REAP', edgecolor = "none")
plt.hist(p3last, 50,  facecolor='orangered', alpha=0.5, label='Single long', edgecolor = "none")

p1last_ave = np.mean(p1last)
p2last_ave = np.mean(p2last)
p3last_ave = np.mean(p3last)

plt.axvline(x=p1last_ave, color='green', lw=2, linestyle='--')
plt.axvline(x=p2last_ave, color="midnightblue", lw=2, linestyle='--')
plt.axvline(x=p3last_ave, color='orangered', lw=2, linestyle='--')

plt.xlabel('Portion of landscape discovered after 15 '+r'$\mu s$')
plt.ylabel('Number of tries')
y = [0, 5, 10, 15]
#ylabel =['0%', '5%', '10%', '15%']
#plt.yticks(y, ylabel)
plt.yticks(y)
plt.xlim([0,190])
x = [0, 95,190]
xlabel = ['0', '0.5', '1']
plt.xticks(x, xlabel)

plt.ylim([0, 15])


plt.legend(fontsize=20, frameon=False, loc=2)
plt.savefig('portion.png', dpi = 300)
plt.show()

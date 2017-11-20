#######
import numpy as np

from scipy import io as sio
import matplotlib.pyplot as plt
zfonsize = 18
font = {'size'   : zfonsize}

plt.rc('font', **font)
plt.rc('xtick', labelsize=zfonsize)
plt.rc('ytick', labelsize=zfonsize)

n_trjs = 100
totalData = 201

fulldata = np.zeros((totalData, 99))

j_index = 0
for j in range(n_trjs):
	try:
		data = np.load('kernel/kernel_LC_'+str(j)+'.npy')
		#print(data)
		for i in range(totalData):
			fulldata[i][j_index] = data[i]
		j_index=j_index+1
	except:
		print('zzz')	

means = np.zeros((totalData, 1))
stds = np.zeros((totalData, 1))
print(fulldata)
for i in range(totalData):
    stds[i] = np.std(fulldata[i])
    means[i] = np.mean(fulldata[i])

print(stds)
np.save('means', np.concatenate(means))
np.save('stds', np.concatenate(stds))

means = np.load('means.npy')
stds = np.load('stds.npy')

xdata = range(totalData)
ydata = means
yerror = stds

print(means, stds)
print(ydata[0])
print(yerror[0])
print(len(ydata+yerror))
color="orangered"
plt.plot(xdata, ydata, color=color)
plt.fill_between(xdata, ydata-yerror, ydata+yerror, alpha=0.3, color=color)
plt.ylim([0,4])
plt.xlabel('K-E distances (nm)')
plt.ylabel('Probability density')
plt.savefig('fig-z.png')
plt.show()

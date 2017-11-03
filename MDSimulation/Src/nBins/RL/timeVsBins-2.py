import numpy as np
import matplotlib.pyplot as plt

fulldata = np.zeros((3000, 50))

jj=0
for j in range(100):
	try:
		data = np.load('n_discoveredS_time'+str(j)+'.npy')
		jj=jj+1

		for i in range(3000):
			print('zz')
			#print(data[i])
			
			fulldata[i][jj] = data[i][1]
	except:
		print(j)

np.save('fulldata', fulldata)

means = np.zeros((3000, 1))
stds = np.zeros((3000, 1))

for i in range(3000):
    stds[i] = np.std(fulldata[i])
    means[i] = np.mean(fulldata[i])


np.save('means', np.concatenate(means))
np.save('stds', np.concatenate(stds))

xdata = range(3000)
ydata = means
yerror = stds
plt.errorbar(xdata, ydata, yerr=yerror)
plt.show()
#plt.errorbar(xdata, ydata, fmt='ro', label="data", xerr=0.75, yerr=yerror, ecolor='black')


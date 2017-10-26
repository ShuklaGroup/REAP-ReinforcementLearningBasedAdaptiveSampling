import numpy as np

fulldata = np.zeros((1000, 100))

for j in range(100):
    data = np.load('n_discoveredS_time'+str(i)+'.npy')
    for i in range(1000):
        fulldata[i][j] = data[j]

np.save('fulldata', fulldata)

means = np.zeros((1000, 1))
stds = np.zeros((1000, 1))

for i in range(1000):
    stds[i] = np.std(fulldata[i])
    means[i] = np.mean(fulldata[i])


np.save('means', means)
np.save('stds', stds)

xdata = np.range(1000)
ydata = means
yerror = stds
plt.errorbar(xdata, ydata, yerr=yerror)
plt.show()
#plt.errorbar(xdata, ydata, fmt='ro', label="data", xerr=0.75, yerr=yerror, ecolor='black')
             
             
             
             
             

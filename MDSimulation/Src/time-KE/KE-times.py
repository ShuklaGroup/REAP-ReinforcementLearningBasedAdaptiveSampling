import numpy as np 

Atimes = []
for i in range(100):
	try:
		trjs_theta = np.load('../data/trjs-theta/trjs_theta' + str(i) + '.npy')
		KE_min = []
		Atime = []
		for frame in range(len(trjs_theta[1])):
			KE_min.append(trjs_theta[1][frame])
			Atime.append(np.min(KE_min))
		Atimes.append(Atime)
	except:
		print('BAD')
np.save('KE-times', Atimes)

frame_n = len(trjs_theta[1])
fulldata = np.zeros((frame_n, 100))
j_index = 0
for j in range(100):
	try:
		data = Atimes[j]
		for i in range(frame_n):
			fulldata[i][j_index] = data[i]
		j_index = j_index+1
	except:
		print('noo')

means = np.zeros((len(trjs_theta[1]), 1))
stds = np.zeros((len(trjs_theta[1]), 1))

for i in range(len(trjs_theta[1])):
    stds[i] = np.std(fulldata[i])
    means[i] = np.mean(fulldata[i])

print(stds)
np.save('means', np.concatenate(means))
np.save('stds', np.concatenate(stds))

"""
import matplotlib.pyplot as plt
#import seaborn

#seaborn.violinplot(Atimes)
fig, axes = plt.subplots(nrows=1, ncols=1)

axes.boxplot([Atimes])

plt.show()
"""


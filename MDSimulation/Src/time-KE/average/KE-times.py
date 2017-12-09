import numpy as np 

Atimes = []
n_trj = 95
for i in range(n_trj):
	try:
		trjs_theta = np.load('../data/trjs-theta/trjs_theta' + str(i) + '.npy')
		KE = []
		KE_min = []
		for frame in range(len(trjs_theta[1])):
			KE.append(trjs_theta[1][frame])
			if frame>1000:
				KE10minMean = np.average(np.sort(KE)[0:1000])
			else:
				KE10minMean = np.average(KE)
			KE_min.append(KE10minMean)
			if KE10minMean==0:
				print(i)

		Atimes.append(KE_min)
	except:
		print('BAD'+str(i))

np.save('KE-times', Atimes)

frame_n = len(trjs_theta[1])
fulldata = np.zeros((frame_n, n_trj))
j_index = 0
for j in range(n_trj):
		data = Atimes[j]
		for i in range(frame_n):
			fulldata[i][j_index] = data[i]
		j_index = j_index+1


means = np.zeros((len(trjs_theta[1]), 1))
stds = np.zeros((len(trjs_theta[1]), 1))

for i in range(len(trjs_theta[1])):
    stds[i] = np.std(fulldata[i])
    means[i] = np.mean(fulldata[i])

print(fulldata)
np.save('fulldata', fulldata)
np.save('means', np.concatenate(means))
np.save('stds', np.concatenate(stds))




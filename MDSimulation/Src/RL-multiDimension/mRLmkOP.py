import numpy as np

ca = np.load('Gens_CA-distances.npy')

r1 = np.random.random_integers(len(ca[0]))
r2 = np.random.random_integers(len(ca[0]))

ca_r1=ca[:,r1]
ca_r2=ca[:,r2]

np.save('Gens_x3', ca_r1)
np.save('Gens_x4', ca_r2)

# r1 = 31123, r2 = 9112


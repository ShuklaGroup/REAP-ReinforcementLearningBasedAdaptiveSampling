
from scipy.stats.kde import gaussian_kde
import numpy as np

r = np.linspace(-np.pi, np.pi,201)

for i in range(11):
	try:
		y = np.load('../../../SL/replicas/rep'+str(i)+'/trjs_theta.npy')[1]
		my_pdf = gaussian_kde(y)
		z=my_pdf(r)
		np.save('kernel'+str(i), z)
	except:
		print('z')


from scipy.stats.kde import gaussian_kde
import numpy as np

r = np.linspace(0,2,201)

for i in range(100):
	try:
		y = np.load('try2-Nov9-2/trjs_theta'+str(i)+'.npy')[1]
		my_pdf = gaussian_kde(y)
		z=my_pdf(r)
		np.save('kernel'+str(i), z)
	except:
		print('z')

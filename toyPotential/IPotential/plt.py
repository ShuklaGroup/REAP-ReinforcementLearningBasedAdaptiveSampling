import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

#x = np.array(trjs_theta[0])
#y = np.array(trjs_theta[1])
plt.rcParams.update({'font.size':18})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)


cdict3 = {'red':  ((0.0, 19/255, 19/255), # blue
                   (0.25, 2*(19/255), 2*(19/255)),
                   (0.5, 0.9, 0.9),
                   (0.75, 238/255, 238/255),  # orange
                   (1.0, 238/255, 238/255)), # orange

         'green': ((0.0, 31/255, 31/255),
                   (0.25, 2*31/255, 2*31/255),
                   (0.5, 0.9, 0.9),
                   (0.75, 83/255, 83/255), # orange
                   (1.0, 83/255, 83/255)), # orange

         'blue':  ((0.0, 51/255, 51/255),
                   (0.25, 3*51/255, 3*51/255),
                   (0.5, 0.9, 0.9),
                   (0.75, 57/255, 57/255), # orange
                   (1.0, 57/255, 57/255)) # orange
        }


cdict3 = {'red':  ((0.0, 238/255, 238/255),  # orange
                #   (0.25, 2*(19/255), 2*(19/255)),
                #   (0.5, 2*238/255, 2*238/255),
                   (0.4, 2*238/255, 2*238/255),
                   (0.85, 19/255, 19/255), # blue
                   (1.0, 1, 1)), # orange

         'green': ((0.0, 83/255, 83/255), # orange
                #   (0.25, 2*31/255, 2*31/255),
                #   (0.5, 2*83/255, 2*83/255),
                   (0.4, 2*83/255, 2*83/255),
                   (0.85, 31/255, 31/255),
                   (1.0, 1, 1)), # orange

         'blue':  ((0.0, 57/255, 57/255), # orange
                   (0.4, 2*57/255, 2*57/255), # orange
                #   (0.25, 2*51/255, 3*51/255),
                #   (0.5, 2*57/255, 2*57/255),
                   (0.85, 51/255, 51/255), # blue
                   (1.0, 1, 1)) # orange
        }

cdict3 = {'red':  ((0.0, 238/255, 238/255),  # orange
                #   (0.25, 2*(19/255), 2*(19/255)),
                #   (0.5, 2*238/255, 2*238/255),
                   (0.4, 1.5*238/255, 1.5*238/255),
                   (0.85, 19/255, 19/255), # blue
                   (1.0, 1, 1)), # orange

         'green': ((0.0, 83/255, 83/255), # orange
                #   (0.25, 2*31/255, 2*31/255),
                #   (0.5, 2*83/255, 2*83/255),
                   (0.4, 1.5*83/255, 1.5*83/255),
                   (0.85, 31/255, 31/255),
                   (1.0, 1, 1)), # orange

         'blue':  ((0.0, 57/255, 57/255), # orange
                   (0.4, 1.5*57/255, 1.5*57/255), # orange
                #   (0.25, 2*51/255, 3*51/255),
                #   (0.5, 2*57/255, 2*57/255),
                   (0.85, 81/255, 81/255), # blue
                   (1.0, 1, 1)) # orange
        }

plt.register_cmap(name='BlueRed3', data=cdict3)
plt.rcParams['image.cmap'] = 'BlueRed3'

# parameters in Mueller potential

aa = [-1.5, -10, -1.5] # inverse radius in x
bb = [0, 0, 0] # radius in xy
cc = [-20, -1, -20] # inverse radius in y
AA = [-80, -80, -80] # strength

XX = [0, 0, 0] # center_x
YY = [0.5, 2, 3.5] # center_y

zxx = np.mgrid[-2:2.01:0.01]
zyy = np.mgrid[0:4.01:0.01]
xx, yy = np.meshgrid(zxx, zyy)


V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
for j in range(1,3):
        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

#fig = plt.figure()
#ax = fig.add_subplot(111)
                
plt.contourf(xx,yy, np.minimum(V1,200), 40, vmin=-80)
#, levels=[-90, -80, -70, -60, -55, -0.05, 0])
#plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
#plt.plot(x, y, 'o')
plt.savefig('fig_all.png')
plt.show()


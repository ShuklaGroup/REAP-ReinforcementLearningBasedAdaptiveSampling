import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
#x = np.array(trjs_theta[0])
#y = np.array(trjs_theta[1])
plt.rcParams.update({'font.size':18})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

cdict3 = {'red':  ((0.0, 238/255, 238/255),  # orange
                   (0.4, 1.5*238/255, 1.5*238/255),
                   (0.85, 19/255, 19/255), # blue
                   (1.0, 1, 1)), # orange

         'green': ((0.0, 83/255, 83/255), # orange
                   (0.4, 1.5*83/255, 1.5*83/255),
                   (0.85, 31/255, 31/255),
                   (1.0, 1, 1)), # orange

         'blue':  ((0.0, 57/255, 57/255), # orange
                   (0.4, 1.5*57/255, 1.5*57/255), # orange
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


trj_x = []
trj_y = []
mu = 8
tol1 = 1e-7
n2 = 1e1;
h = 1e-4
inits_x = [-1,0, 0,1, -1,0, 0,1, 0,0,0,0,0.2,0.3,-0.2,0, 0,0, 0, 0,0,0,0,0.2,0.3,-0.2]
inits_y = [0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.6, 0.5, 0.7, 0.8,0.9, 0.8,0.9, 0.8,0.9, 1.5, 1.5, 1.6, 1.5, 1.5, 1.5, 1.6, 1.5, 1.7, 1.8,1.9]
x = np.array(inits_x)
y = np.array(inits_y)
nstepmax = 500
n1 = len(inits_x)

for nstep in range(int(nstepmax)):
        
# calculation of the x and y-components of the force, dVx and dVy respectively

        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
        dVx = (aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
        dVy = (bb[0]*(x-XX[0])+cc[0]*(y-YY[0]))*ee

        for j in range(1,3):
                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                dVx = dVx + (aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                dVy = dVy + (bb[j]*(x-XX[j])+cc[j]*(y-YY[j]))*ee

        x0 = x
        y0 = y
        x = x - h*dVx + np.sqrt(h*mu)*np.random.randn(1,n1)
        y = y - h*dVy + np.sqrt(h*mu)*np.random.randn(1,n1)

        trj_x.append(x[0]) 
        trj_y.append(y[0])

 

plt.contourf(xx,yy, np.minimum(V1,200), 40, vmin=-80)
#, levels=[-90, -80, -70, -60, -55, -0.05, 0])
#plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
print(len(trj_x))
plt.plot(trj_x, trj_y, 'o', markersize=8)
np.save('trj_x', trj_x)
np.save('trj_y', trj_y)
plt.savefig('fig_all_withTrj.png')
plt.show()

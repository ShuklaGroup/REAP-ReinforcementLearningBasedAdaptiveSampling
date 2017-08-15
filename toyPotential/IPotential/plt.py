import numpy as np
import matplotlib.pyplot as plt
#x = np.array(trjs_theta[0])
#y = np.array(trjs_theta[1])

# parameters in Mueller potential

aa = [-2, -20, -2] # inverse radius in x
bb = [0, 0, 0] # radius in xy
cc = [-20, -2, -20] # inverse radius in y
AA = 30*[-80, -80, -80] # strength

XX = [0, 0, 0] # center_x
YY = [1, 2, 3] # center_y

zxx = np.mgrid[-1:4.1:0.01]
zyy = np.mgrid[-1:4.1:0.01]
xx, yy = np.meshgrid(zxx, zyy)


V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
for j in range(1,3):
        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

fig = plt.figure()
ax = fig.add_subplot(111)
                
ax.contourf(xx,yy,np.minimum(V1,200), 40)

plt.xlabel('x')
plt.ylabel('y')
#plt.plot(x, y, 'o')
plt.savefig('fig_all.png')

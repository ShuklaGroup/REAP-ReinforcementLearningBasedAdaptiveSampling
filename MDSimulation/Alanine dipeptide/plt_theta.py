import matplotlib.pyplot as plt
import numpy as np
figName = 'fig.png'
plt.rcParams.update({'font.size':30})
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)


#w = np.load('trjs_theta.npy')

phi = np.load('phi.npy')
psi = np.load('psi.npy')

#plt.scatter(w[0], w[1])
print(len(phi))

#plt.scatter(phi, psi, color='dodgerblue', s=3, alpha=0.2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(phi, psi, color='darkorange', s=5, alpha=0.2)

"""
plt.xlim([-200, 200])
plt.xticks(np.arange(-200, 300, 100))
plt.ylim([-200, 200])
plt.yticks(np.arange(-200, 300, 100))
"""

plt.xlim([-180, 180])
plt.xticks([-180, -90, 0, 90, 180])

ax.set_yticklabels([r'-$\pi$', r'-$\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'])
ax.set_xticklabels([r'-$\pi$', r'-$\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'])

plt.ylim([-180, 180])
plt.yticks([-180, -90, 0, 90, 180])


plt.xlabel(r'$\phi$')
plt.ylabel(r'$\psi$')
fig.savefig(figName, dpi=1000, bbox_inches='tight')
plt.show()

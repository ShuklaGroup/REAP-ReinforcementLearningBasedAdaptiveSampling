import numpy as np
import matplotlib.pyplot as plt


means = np.load('means.npy')
stds = np.load('stds.npy')

xdata = range(3000)

ydata = means

yerror = stds


print(ydata[0])
print(yerror[0])
print(len(ydata+yerror))
#plt.errorbar(xdata, ydata, yerr=yerror)
#plt.errorbar(xdata, ydata, fmt='ro', label="data", xerr=0.75, yerr=yerror, ecolor='black')
#color = ax._get_lines.color_cycle.next()
color="orangered"
plt.plot(xdata, ydata, color=color)
plt.fill_between(xdata, ydata-yerror, ydata+yerror, alpha=0.3, color=color)
plt.show()

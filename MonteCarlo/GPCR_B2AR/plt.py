
import numpy as np
import RLSim as rl

stepS = 0.001*0.001*50 # milli Second B2AR
times = np.loadtxt('times.txt')
times2 = stepS * times
rl.pltTimes(times2)

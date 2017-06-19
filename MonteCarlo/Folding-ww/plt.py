import numpy as np
import RLSim as rl

stepS = 0.001*120 # microSecond WW
times = np.loadtxt('times.txt')
times2 = stepS * times
rl.pltTimes_folding(times2)

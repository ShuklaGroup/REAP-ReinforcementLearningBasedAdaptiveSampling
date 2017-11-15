import LCSim as lc

z = lc.mockSimulation()

# 2 ns
z.runSimulation(R=1000, N=1, s=1000, method='LC')


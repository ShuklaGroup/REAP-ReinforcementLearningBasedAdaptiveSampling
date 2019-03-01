production_steps = 1000000 total simulation # equivalent to 2 ns of production MD 

rounds = 100 

each round steps = 10000 # equivalent to 0.02 ns (20 pico s) of production MD 

```
import RLSim as rl
z = rl.mockSimulation()
z.runSimulation(R=99, s=10000)
```

```
import numpy as np
d = np.loadtxt('Mapping.dat')
map2 = []
for i in range(2000):
    if int(d[i])!=-1:
        map2.append(i)     
```

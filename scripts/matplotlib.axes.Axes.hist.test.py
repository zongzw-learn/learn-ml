import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(10000) 
fig, ax = plt.subplots()
  
n, bins, patches = ax.hist(x, 100) 

plt.show()
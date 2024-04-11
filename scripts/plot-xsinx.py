import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot()

x = np.arange(0, 200, 1)
y = x * np.sin(x)

ax.plot(x, y)

plt.show()
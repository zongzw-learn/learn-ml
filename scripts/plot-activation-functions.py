import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2)

x = np.arange(-5, 5, 0.05)
y = 1/(1+np.exp(-x))

axes[0, 0].plot(x, y)
axes[0, 0].set_title('sigmoid')

y = (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
axes[0, 1].plot(x, y)
axes[0, 1].set_title('tanh')

x = np.arange(-5, 0, 0.05)
y = np.zeros(len(x))
axes[1, 0].plot(x, y)
x = np.arange(0, 5, 0.05)
y = x
axes[1, 0].plot(x, y)
axes[1, 0].set_title('ReLu')

plt.show()
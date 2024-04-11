import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.arange(-10, 10, 1)
y = np.arange(-10, 10, 1)

x, y = np.meshgrid(x, y)
z = x*y
# z = x*x + y*y
# ax.plot_wireframe(x, y, z)
ax.plot_surface(x, y, z)

# ax.scatter3D(x, y, z, c=z, cmap='cividis');

plt.show()
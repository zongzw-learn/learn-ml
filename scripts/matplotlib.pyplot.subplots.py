import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801) 
  
xdata = np.random.random([3, 10]) 
  
xdata1 = xdata[0, :] 
xdata2 = xdata[1, :] 
xdata3 = xdata[2, :]
  
ydata1 = xdata1 ** 2
ydata2 = 1 - xdata2 ** 3
ydata3 = np.random.random([10])

fig = plt.figure() 
ax = fig.add_subplot() 
#ax = fig.add_subplot(4, 2, 5) 
ax.plot(xdata1, ydata1, color ='tab:blue') 
ax.plot(xdata2, ydata2, color ='tab:orange') 
ax.scatter(xdata3, ydata3, color ='tab:red') 
ax.plot(xdata3, ydata3, color ='tab:red') 
   
#ax.set_xlim([0, 1]) 
#ax.set_ylim([0, 1]) 
fig.suptitle('matplotlib.figure.Figure.add_subplot() function Example\n\n', fontweight ="bold") 
  
plt.show() 

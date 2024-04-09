import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10**7) 
mu = 121  
sigma = 21
x = mu + sigma * np.random.randn(1000) 
  
num_bins = 100
fig, ax = plt.subplots() 
  
n, bins, patches = ax.hist(x, num_bins, 
                           density = 1,  
                           color ='green',  
                           alpha = 0.7) 
  
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2)) 
ax.plot(bins, y, '--', color ='black') 
ax.set_xlabel('X-Axis') 
ax.set_ylabel('Y-Axis') 
  
ax.set_title('matplotlib.axes.Axes.hist() Example') 
plt.show() 
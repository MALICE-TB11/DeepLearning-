import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return np.array(x > 0, dtype = int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = step(x)
z = sigmoid(x)
a = ReLU(x)

plt.plot(x, y)
plt.plot(x, z)
plt.plot(x, a)

plt.ylim(-0.1, 1.1)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.array([10,20,30,40,50])
y = np.array([10,20,30,40,50])

X, Y = np.meshgrid(x, y)

Z = np.minimum(X, Y) / np.maximum(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='jet')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Membership')

plt.show()
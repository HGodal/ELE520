import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-5, 7, N)
Y = np.linspace(-5, 7, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([1, 1])
Sigma = np.array([[ 5, 3], [3, 5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

F = multivariate_normal(mu, Sigma)
Z = F.pdf(pos)

# Create a surface plot and projected filled contour plot under it.
#fig = plt.figure()
#ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
#                 cmap=cm.viridis)

fig,ax=plt.subplots(1,1)

cset = ax.contourf(X, Y, Z) #, zdir='z', offset=0)


plt.plot(1, 1, 'bo')
ax.arrow(1.0, 1.0, 2.0, 2.0, head_width=0.2, head_length=0.3, color='k')
ax.arrow(1.0, 1.0, -1.0, 1.0, head_width=0.2, head_length=0.3, color='k')
# Adjust the limits, ticks and view angle
# ax.set_zlim(0,0.1)
# ax.set_zticks(np.linspace(0,0.1,2))
# ax.view_init(27, -21)

plt.show()
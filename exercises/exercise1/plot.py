from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def cuboid_data(o, size=(1, 1, 1)):
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('auto')

X, Y, Z = cuboid_data((1, 1, 0), (1, 1, 1))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=2,
                antialiased=True, alpha=0.3, color='yellow')

plt.xlim([0, 3])
plt.ylim([0, 3])
ax.set_zlim(0, 3)

ax.plot([1.5], [1.5], [1.01], markerfacecolor='k',
        markeredgecolor='k', marker='o', markersize=5)
ax.plot([3/2, 3/2], [3/2, 3/2], [0, 3])


plt.show()

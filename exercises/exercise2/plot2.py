import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math

N = 600
X = np.linspace(-2, 8, N)
a = np.linspace(-2, 8, N)
Y = np.linspace(-4, 6, N)
X, Y = np.meshgrid(X, Y)

border = []
for i in range(len(a)):
    border.append(3/10 * a[i]*a[i] - 9/5 * a[i] + 2.923)

# Mean vector and covariance matrix
mu1 = np.array([3, 3])
sigma1 = np.array([[0.5, 0], [0, 2]])

mu2 = np.array([3, -2])
sigma2 = np.array([[2, 0], [0, 2]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

F1 = multivariate_normal(mu1, sigma1)
F2 = multivariate_normal(mu2, sigma2)
Z1 = F1.pdf(pos)
Z2 = F2.pdf(pos)
Z = (Z1 - Z2) * 2

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)

plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
plt.text(2.05, 5, 'Distribution 1')
plt.text(2.05, -3.8, 'Distribution 2')

plt.clabel(CS, inline=1, fontsize=6)
plt.plot(a, border, color='red')
plt.text(5, 1, 'Decision border')
plt.ylim(-4, 6)

ax.arrow(3.0, 3.0, (math.sqrt(1/2)), 0.0, width=0.00001,
         head_width=0.1, head_length=0.2, color='k')
ax.arrow(3.0, 3.0, 0.0, math.sqrt(2), width=0.00001,
         head_width=0.1, head_length=0.2, color='k')
ax.arrow(3.0, -2.0, (math.sqrt(2)), 0.0, width=0.00001,
         head_width=0.1, head_length=0.2, color='k')
ax.arrow(3.0, -2.0, 0.0, math.sqrt(2), width=0.00001,
         head_width=0.1, head_length=0.2, color='k')
ax.set_aspect('equal')
plt.show()

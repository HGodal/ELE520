import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

def ellipse(mu, eigvec, col, ax, plotArrows, style):
    rotation = math.degrees(math.atan((eigvec[0, 1]) / (eigvec[0, 0])))

    newEllipse = Ellipse((mu[0], mu[1]),
                         width=2*math.sqrt(eigvec[0, 0]**2 + eigvec[0, 1]**2),
                         height=2*math.sqrt(eigvec[1, 0]**2 + eigvec[1, 1]**2),
                         facecolor='none',
                         edgecolor=col,
                         linestyle=style,
                         angle=rotation)

    ax.add_patch(newEllipse)
    if plotArrows:
        ax.arrow(mu[0], mu[1], eigvec[1, 0],
                 eigvec[1, 1], width=0.03, color='black')
        ax.arrow(mu[0], mu[1], eigvec[0, 0],
                 eigvec[0, 1], width=0.03, color='black')

# From exercise 3
mu1 = np.array([3, 6])
mu2 = np.array([3, -2])
sigma1 = np.array([[0.5, 0], [0, 2]])
sigma2 = np.array([[2.045, 0.3], [0.3, 2]])

eigval1, eigvec1 = np.linalg.eig(sigma1)
eigval2, eigvec2 = np.linalg.eig(sigma2)
eigvec1 = eigval1*eigvec1
eigvec2 = eigval2*eigvec2

# From exercise 2
mu3 = np.array([3, 3])
mu4 = np.array([3, -2])
sigma3 = np.array([[0.5, 0], [0, 2]])
sigma4 = np.array([[2, 0], [0, 2]])

eigval3, eigvec3 = np.linalg.eig(sigma3)
eigval4, eigvec4 = np.linalg.eig(sigma4)
eigvec3 = eigval3*eigvec3
eigvec4 = eigval4*eigvec4


fig, ax = plt.subplots()
plt.xlim(-5, 10)
plt.ylim(-5, 10)

ellipse(mu1, eigvec1, 'orange', ax, False, '-')
ellipse(mu2, eigvec2, 'blue', ax, False, '-')
ellipse(mu3, eigvec3, 'green', ax, False, '--')
ellipse(mu4, eigvec4, 'red', ax, False, '--')

xRange = np.linspace(-5, 10, 1000)

border1 = []
border2 = []
for i in range(len(xRange)):
    border1.append(3/10 * xRange[i]*xRange[i] - 9/5 * xRange[i] + 2.923)
    border2.append(
        (-4.248 + 0.075*xRange[i] + math.sqrt(0.0225*xRange[i]*xRange[i] - 0.732825*xRange[i]+18.35103))/(0.01125))
plt.plot(xRange, border1, color='red', linewidth=0.5, linestyle='--')
plt.plot(xRange, border2, color='purple', linewidth=0.5, linestyle='-')

fig.gca().set_aspect('equal')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Bivariate gaussians')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

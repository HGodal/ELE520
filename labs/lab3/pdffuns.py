import numpy as np
import matplotlib.pyplot as plt


def norm1D(my, Sgm, x):
    p = np.zeros(np.shape(x))
    n, _ = np.shape(x)

    const = 1 / (np.sqrt(2 * np.pi) * Sgm)

    for i in range(n):
        p[i] = const * \
            np.exp(-1 / 2 * np.square((x[i] - my)) / (np.square(Sgm)))

    return p


def norm2D(my, sigma, X1, X2):
    p = np.zeros(np.shape(X1))
    dim1, dim2 = np.shape(X1)

    const = (2*np.pi)**(-len(my)/2) * (np.linalg.det(sigma))**(-1/2)

    for i in range(dim1):
        for j in range(dim2):
            x_mu = ([X1[i, j], X2[i, j]] - my)
            p[i, j] = const * \
                np.exp(-1/2 * np.linalg.multi_dot([
                    x_mu.T,
                    np.linalg.inv(sigma),
                    x_mu
                ]))
    return p


def knn2D(data, kn, X1, X2):
    p = np.zeros(np.shape(X1))
    dim1, dim2 = np.shape(X1)

    n = len(data[0])
    distances = []

    for i in range(n):
        my = data[:, i]
        distances.append(np.sqrt((X1 - my[0])**2 + (X2 - my[1])**2))

    for i in range(dim1):
        for j in range(dim2):
            distpoints = [distance[i][j] for distance in distances]
            idx = np.argsort(distpoints, 0)
            r = distpoints[idx[kn-1]]
            V = np.pi * r**2
            p[i, j] = kn / (n*V)

    return p


def classplot(every_g, X1, X2, overlappings=[True, False], title=''):
    fig = plt.figure(figsize=plt.figaspect(1/2))
    fig.suptitle(title)
    titles = [r'$P(\omega_i)p(\mathbf{x}|\omega_i)$',
              r'$P(\omega_i | \mathbf{x})$']

    for i, g in enumerate(every_g, 1):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.elev = 20
        ax.azim = -55
        ax.set_title(titles[i-1])

        for j in range(len(g)):
            new_g = np.copy(g[j])
            if overlappings[i-1]:
                overlap_mask = (new_g >= np.max(g, axis=0)).astype(float)
                np.putmask(new_g, overlap_mask == 0, np.nan)

            ax.plot_surface(X1, X2, new_g, facecolor=['r', 'b'])
        ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$')

    plt.show()

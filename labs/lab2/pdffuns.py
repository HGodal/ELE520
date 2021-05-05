import numpy as np
import matplotlib.pyplot as plt


def norm1D(my, Sgm, x):
    [n, _] = np.shape(x)
    p = np.zeros(np.shape(x))
    for i in np.arange(0, n):
        p[i] = 1 / (np.sqrt(2 * np.pi) * Sgm) * \
            np.exp(-1 / 2 * np.square((x[i] - my)) / (np.square(Sgm)))

    return p


def norm2D(mu, sigma, X1, X2):
    p = np.zeros(np.shape(X1))
    dim1, dim2 = np.shape(X1)

    const = (2*np.pi)**(-len(mu)/2) * (np.linalg.det(sigma))**(-1/2)

    for i in np.arange(0, dim1):
        for j in np.arange(0, dim2):
            x_mu = ([X1[i, j], X2[i, j]] - mu)
            p[i, j] = const * \
                np.exp(-1/2 * np.linalg.multi_dot([
                    x_mu.T,
                    np.linalg.inv(sigma),
                    x_mu
                ]))
    return p


def normplot(every_g, X1, X2, overlapping=False, title=''):
    fig = plt.figure(figsize=plt.figaspect(1/2))
    fig.suptitle(title)

    for i, g in enumerate(every_g, 1):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.elev = 25
        ax.azim = -55

        if i == 1:
            ax.set_title('Prior probability 0.5 and 0.5')
        elif i == 2:
            ax.set_title('Prior probability 0.1 and 0.9')

        for i in range(len(g)):
            new_g = np.copy(g[i])
            if overlapping:
                overlap_mask = (new_g >= np.max(g, axis=0)).astype(float)
                np.putmask(new_g, overlap_mask == 0, np.nan)

            ax.plot_surface(X1, X2, new_g, facecolor=['r', 'b'])
        ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$')

    plt.show()

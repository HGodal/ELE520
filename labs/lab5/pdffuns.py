import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.color_palette('bright')
plt.rcParams['figure.figsize'] = [12, 5]


def separate_classes(x, y):
    x_train = np.vstack([_.T for _ in x])
    x_test = np.vstack([_.T for _ in y])

    y_train = np.array([i+1 for i, data in enumerate(x)
                        for _ in range(data.shape[1])])
    y_test = np.array([i+1 for i, data in enumerate(y)
                       for _ in range(data.shape[1])])

    return x_train, x_test, y_train, y_test


def accuracy_score(predicted_labels, true_labels):
    cm = confusion_matrix(predicted_labels, true_labels, normalize=False)
    return cm.trace() / cm.sum()


def error_score(predicted_labels, true_labels):
    return 1 - accuracy_score(predicted_labels, true_labels)


def plot_data_2D(train, test, title):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title, fontsize=16)
    for i, dataset in enumerate([train, test]):
        ax[i].set(title='Training data' if i == 0 else 'Test data')
        for j, data in enumerate(dataset):
            ax[i].scatter(data[0], data[1], label=fr'$\omega_{j+1}$')
    plt.legend()


def plot_data_3D(train, test, title):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(title, fontsize=16)
    for i, dataset in enumerate([train, test]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.set(title='Training data' if i == 0 else 'Test data')
        for j, data in enumerate(dataset):
            ax.scatter(data[0], data[1], data[2], label=fr'$\omega_{j+1}$')
    plt.legend()


def confusion_matrix(predicted_labels, true_labels, normalize=True):
    matshape = len(np.unique(true_labels))
    cm = np.zeros((matshape, matshape))

    for i in range(len(predicted_labels)):
        cm[predicted_labels[i]-1, true_labels[i]-1] += 1

    return cm / cm.sum(axis=0) if normalize else cm

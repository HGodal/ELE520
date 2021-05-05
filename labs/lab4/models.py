import numpy as np


class MLClassifier:
    def __init__(self):
        self.classes = []
        self.prior = {}
        self.mu = {}
        self.sgm = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            class_idx = np.where(y == label)
            self.classes.append(label)
            self.prior[label] = np.size(class_idx) / np.size(y)
            self.mu[label] = np.mean(x[class_idx], axis=0)
            self.sgm[label] = np.cov(x[class_idx], rowvar=False)

    def predict(self, x):
        predictions = []
        for point in x:
            discrs = {}
            for classnum in self.classes:
                discrs[classnum] = self._discriminant(point, classnum)

            predictions.append(max(discrs, key=discrs.get))

        return np.array(predictions)

    def _discriminant(self, x, label):
        # Case 3 discriminant function
        x_mu = x - self.mu[label]
        sgminv = np.linalg.inv(self.sgm[label])
        sgmdet = np.linalg.det(self.sgm[label])

        p = -1/2 * np.linalg.multi_dot([x_mu.T, sgminv, x_mu]) - \
            1/2 * np.log(sgmdet) + np.log(self.prior[label])
        return p


class ParzenClassifer:
    def __init__(self, h1):
        self.h1 = h1
        self.classes = []
        self.points = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            class_idx = np.where(y == label)
            self.classes.append(label)
            self.points[label] = x[class_idx]

    def predict(self, x):
        predictions = []
        for point in x:
            discrs = {}
            for classnum in self.classes:
                discrs[classnum] = self._discriminant(point, classnum)

            predictions.append(max(discrs, key=discrs.get))

        return np.array(predictions)

    def _discriminant(self, x, label):
        # Normal parzen window
        dim = self.points[label].shape[-1]
        N = len(self.points[label])
        hn = self.h1 / np.sqrt(N)

        all_u = np.array([(x-x_i) / hn for x_i in self.points[label]])
        const = 1 / (N * hn**dim * (2*np.pi)**(dim/2) * 1**(dim/2))

        p = const * sum([np.exp(-1/2*np.dot(u, u)) for u in all_u])

        return p


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.classes = []
        self.points = {}

    def fit(self, x, y):
        class_labels = np.unique(y)
        for label in class_labels:
            class_idx = np.where(y == label)
            self.classes.append(label)
            self.points[label] = x[class_idx]

    def predict(self, x):
        prediction = []

        for point in x:
            classdist = {}

            for c in self.classes:
                classdist[c] = sum(self._closest_k_euclidian(point, c))

            prediction.append(min(classdist, key=classdist.get))

        return np.array(prediction)

    def _closest_k_euclidian(self, x, c):
        return sorted([np.linalg.norm(p - x, ord=2) for p in self.points[c]])[:self.k]

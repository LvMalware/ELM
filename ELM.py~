import numpy as np

def mtf_rbf(ax_b):
    return np.exp(-((ax_b * 0.5) ** 2))

def sigmoid(ax_b):
    return 1.0 / (1.0 + np.exp(-ax_b))

def fourier(x):
    return np.sin(-x)

class ELM:
    def __init__(self, n_input, n_hidden, n_output, func=0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.g_func = [ sigmoid, fourier, mtf_rbf ][func]
        self._bias = None
        self._weights = None
        self._beta = None

    def beta(self, new=None):
        if new is not None:
            self._beta = new
        return self._beta

    def bias(self, new=None):
        if new is not None:
            self._bias = new
        return self._bias

    def weights(self, new=None):
        if new is not None:
            self._weights = new
        return self._weights

    def init_params(self):
        self.weights(np.random.uniform(-1, 1, size=(self.n_input, self.n_hidden)))
        self.bias(np.random.uniform(0, 1, size=(self.n_hidden,)))
        return self

    def fit(self, x, y):
        if self._weights is None:
            self.init_params()
        if len(y.shape) == 1:
            y = np.c_[y, [1 - i for i in y]]
        h = self.g_func(x.dot(self.weights()) + self.bias())
        pinv = np.linalg.pinv(h)
        self.beta(pinv.dot(y))
        return self

    def predict(self, x):
        h = self.g_func(x.dot(self.weights()) + self.bias())
        res = h.dot(self.beta())
        if res.shape[1] == 2:
            res = [ 1 if y[0] > y[1] else 0 for y in res.tolist() ]
        return res

    def score(self, x, y):
        pred = self.predict(x)
        success = 0
        for i in range(len(y)):
            if y[i] == pred[i]:
                success += 1
        return success / len(y)

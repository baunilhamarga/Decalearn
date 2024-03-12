# Author: Artur Jordao <arturjlcorreia@gmail.com>

import numpy as np
from scipy import linalg
from scipy import stats

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.model_selection import train_test_split

class infFS_U():
    """.

    Parameters
    ----------
    copy : bool, (default=True)
        If False, X will be overwritten. ``copy=False`` can be used to
        save memory but is unsafe for general use.

    Attributes
    ----------

    References
    ----------
    ""G. Roffo et al A Graph-based Feature Filtering Approach. In the IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). In PAMI, 2020"""

    def __init__(self, alpha=0.5, top_n=10, verbose=False, copy=True):
        self.__name__ = 'Infinite Feature Selection Unsupervised'
        self.copy = copy
        self.verbose = verbose
        self.alpha = alpha
        self.top_n = top_n

    def bsxfun(self, func, A, B):
        dim = A.shape[0]
        output = np.zeros((dim, dim))
        for i in range(0, dim):
            for j in range(0, dim):
                output[i][j] = func(A[i], B[j])

        return output

    def fit(self, X, Y=None):
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        #1) Standard Deviation over the samples

        corr_ij, _ = stats.spearmanr(X)
        corr_ij[np.where(np.isnan(corr_ij))[0]] = 0 # remove NaN
        corr_ij[np.where(np.isinf(corr_ij))[0]] = 0 # remove inf
        corr_ij = 1-np.abs(corr_ij)

        STD = np.std(X, axis=0, ddof=1)
        STDMatrix = self.bsxfun(max, STD, STD.T)
        STDMatrix = STDMatrix - np.min(STDMatrix)
        sigma_ij = STDMatrix / np.max(STDMatrix)
        sigma_ij[np.where(np.isnan(sigma_ij))[0]] = 0 # remove NaN
        sigma_ij[np.where(np.isinf(sigma_ij))[0]] = 0 # remove inf

        #2) Building the graph G = <V,E>
        if self.verbose:
            print('2) Building the graph G = <V,E>')

        N = X.shape[1]
        eps = 5e-06 * N
        factor = 1 - eps # shrinking

        A = (self.alpha*sigma_ij + (1-self.alpha)*corr_ij)

        rho = np.max(np.sum(A, axis=1))

        # Substochastic Rescaling
        A = A / (np.max(np.sum(A, axis=1))+eps)

        # Letting paths tend to infinite: Inf-FS Core
        I = np.eye((A.shape[0])) #Identity Matrix

        r = factor/rho

        y = I - (r * A)

        S = np.linalg.inv(y)

        #5) Estimating energy scores
        self.WEIGHT = np.sum(S, axis=1) # prob. scores s(i)

        #6) Ranking features according to s
        self.RANKED = np.argsort(self.WEIGHT, axis=-1)[::-1]

        e = np.ones((N))
        t = np.dot(S, e)

        nbins = int(0.5 * N)
        counts = np.histogram(t, nbins)[0]

        thr = np.mean(counts)
        size_sub = np.sum(counts>thr)
        self.subset = self.RANKED[0:size_sub]
        return self

    def transform(self, X, Y=None, top_n=None):
        #Update the how many features we wanna w.r.t its weights
        if top_n is not None:
            self.top_n = top_n

        return X[:, self.RANKED[0:self.top_n]]

if __name__ == '__main__':
    np.random.seed(12227)

    n_samples, n_classes = 19000, 19
    X = np.random.rand(n_samples, 100)
    y = np.random.randint(0, n_classes, size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    fs = infFS_U()
    fs.fit(X, y)
    print(fs.RANKED)
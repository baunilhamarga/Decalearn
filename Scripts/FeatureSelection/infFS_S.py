# Author: Artur Jordao <arturjlcorreia@gmail.com>

import numpy as np
from numpy import matlib
from scipy import linalg
from scipy import stats

from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.model_selection import train_test_split

class infFS_S():
    """Inf-FS Supervised (2020).

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

    def __init__(self, alpha=[0.5, 0.5, 0.5], top_n=10, verbose=False, copy=True):
        self.__name__ = 'Infinite Feature Selection Supervised'
        self.copy = copy
        self.verbose = verbose
        self.top_n = top_n
        self.alpha = alpha

    def bsxfun(self, func, A, B):
        dim = A.shape[0]
        output = np.zeros((dim, dim))
        for i in range(0, dim):
            for j in range(0, dim):
                output[i][j] = func(A[i], B[j])

        return output

    def muteinf(self, A, Y):
        n = A.shape[0]
        Z = np.column_stack((A, Y[:, 0]))
        if n/10 > 20:
            nbins = 20
        else:
            nbins = int(max(np.floor(n/10), 10))

        pA = np.histogram(A, nbins)[0]
        pA = pA/n
        pA[np.where(pA==0)[0]] = 0.00001
        od = Y.shape[1]
        c_l = od
        if od == 1:
            pY = np.array([len(np.where(Y==+1.0)[0]),  len(np.where(Y==+-1.0)[0])])/n
            c_l = 2
        else:
            pY = np.zeros((1, od))
            for i in range(0, od):
                pY[i] = len(np.where(Y==+1.0)[0])
            pY = pY / n

        p = np.zeros((c_l, nbins))
        rx = np.abs(np.max(A)-np.min(A))/nbins

        for i in range(0, c_l):
            x_1 = np.min(A)
            for j in range(0, nbins):
                if i == 1 and od == 1:
                    interval = ((x_1 <= Z[:, 0]) & (Z[:, 1] == -1.0))*1
                else:
                    interval = ((x_1 <= Z[:, 0]) & (Z[:, 1] == +1.0))*1
                if j < nbins-1:
                    interval = interval & (Z[:, 0] < x_1 + rx)

                p[i, j] = len(np.where(interval==1)[0])

                if p[i, j] == 0:
                    p[i, j] = 0.00001

                x_1 = x_1 + rx

        HA = -np.sum(pA * np.log(pA))
        HY = -np.sum(pY * np.log(pY))
        pA = matlib.repmat(pA, c_l, 1)
        pY = matlib.repmat(np.reshape(pY, (1, 2)).T, 1, nbins)
        p = p/n
        info = np.sum(np.sum(p * np.log(p /(pA * pY)), axis=0))
        info = 2 * info / (HA + HY)
        return info

    def getGraphWeights(self, train_x, train_y, alpha, eps):

        # Metric 1: Mutual Information
        mi_s = []

        for i in range(0, train_x.shape[1]):
            mi_s.append(self.muteinf(train_x[:, i], train_y))
        mi_s = np.array(mi_s)
        mi_s[np.where(np.isnan(mi_s))[0]] = 0 # remove NaN
        mi_s[np.where(np.isinf(mi_s))[0]] = 0 # remove inf


        #Zero-Max norm
        mi_s = mi_s - np.min(np.min(mi_s))
        mi_s = mi_s / np.max(np.max(mi_s))

        #Metric 2: class separation
        fi_s = np.array([np.mean(train_x[np.where(train_y == +1.0)[0], :], axis=0) - np.mean(train_x[np.where(train_y == -1.0)[0], :], axis=0)])**2
        st = np.std(train_x[np.where(train_y == +1.0)[0],:], axis=0, ddof=1)**2
        st = st + np.std(train_x[np.where(train_y == -1.0)[0], :], axis=0, ddof=1)**2
        st[np.where(st == 0)[0]] = 10000 # remove ones where nothing occurs
        fi_s = fi_s / st

        fi_s[np.where(np.isnan(fi_s))[0]] = 0  # remove NaN
        fi_s[np.where(np.isinf(fi_s))[0]] = 0  # remove inf

        #Zero-Max norm
        fi_s = fi_s - np.min(np.min(fi_s))
        fi_s = fi_s / np.max(np.max(fi_s))

        std_s = np.std(train_x, axis=0, ddof=1)
        std_s[np.where(np.isnan(std_s))[0]] = 0  # remove NaN
        std_s[np.where(np.isinf(std_s))[0]] = 0  # remove inf

        SD = self.bsxfun(max, std_s, std_s.T)
        SD = SD - np.min(np.min(SD))
        SD = SD / np.max(np.max(SD))

        mi_s = np.reshape(mi_s, (1, mi_s.shape[0]))
        MI = np.matlib.repmat(mi_s, mi_s.shape[1], 1)
        FI = np.matlib.repmat(fi_s, fi_s.shape[1], 1)

        G = alpha[0] * ((MI + FI.T)/2) + alpha[1] * ( (MI+SD.T)/2 ) + alpha[2]* ( (SD+FI.T)/2 )

        rho = np.max(np.sum(G, axis=1))

        # Substochastic Rescaling
        G = G / (np.max(np.sum(G, axis=1)) + eps)
        return G, rho

    def fit(self, X, Y):
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        Y = check_array(Y, copy=self.copy, dtype=FLOAT_DTYPES, ensure_2d=False)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y[np.where(Y == 0)[0]] = -1

        N = X.shape[1]

        eps = 5e-06 * N

        factor = 1 - eps # shrinking

        A, rho = self.getGraphWeights(X, Y, self.alpha, eps)

        # Letting paths tend to infinite: Inf-FS Core
        I = np.eye((A.shape[0]))

        r = factor / rho # Set a meaningful value for r: = 0 < r < 1 / rho

        y = I - (r*A)

        S = np.linalg.inv(y)

        # 5) Estimating energy scores
        self.WEIGHT = np.sum(S, axis=1)# prob.scores s(i)

        #6) Ranking features according to s
        self.RANKED = np.argsort(self.WEIGHT, axis=-1)[::-1]

        e = np.ones((N))
        t = np.dot(S, e)

        nbins = int(0.5 * N)
        counts = np.histogram(t, nbins)[0]

        thr = np.mean(counts)
        size_sub = np.sum(counts > thr)
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

    fs = infFS_S()
    fs.fit(X, y)
    print(fs.RANKED)
import numpy as np
from scipy import stats
import random
import math

class GMM(object):
    def __init__(self, X, k=2):
        # dimension
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        # number of mixtures
        self.k = k

    def _init(self):
        # init mixture means/sigmas
        self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.phi = np.ones(self.k) / self.k
        self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        # print(self.mean_arr)
        # print(self.sigma_arr)

    def fit(self, tol=1e-5):
        self._init()
        num_iters = 0
        ll = 1
        previous_ll = 0
        while (ll - previous_ll > tol):
            previous_ll = self.loglikelihood()
            self._fit()
            num_iters += 1
            ll = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f' % (num_iters, ll))
        print('Terminate at %d-th iteration:log-likelihood is %.6f' % (num_iters, ll))

    def loglikelihood(self):
        ll = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                # print(self.sigma_arr[j])
                tmp += stats.multivariate_normal.pdf(self.data[i, :],
                                                     self.mean_arr[j, :].A1,
                                                     self.sigma_arr[j, :]) * \
                       self.phi[j]
            ll += np.log(tmp)
        return ll

    def _fit(self):
        self.e_step()
        self.m_step()

    def e_step(self):
        # calculate w_j^{(i)}
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = stats.multivariate_normal.pdf(self.data[i, :],
                                                    self.mean_arr[j].A1,
                                                    self.sigma_arr[j]) * \
                      self.phi[j]
                den += num
                self.w[i, j] = num
            self.w[i, :] /= den
            assert self.w[i, :].sum() - 1 < 1e-4

    def m_step(self):
        for j in range(self.k):
            const = self.w[:, j].sum()
            self.phi[j] = 1 / self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.w[i, j])
                _sigma_j += self.w[i, j] * (
                (self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
                # print((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const
            # print(self.sigma_arr)


def gauss(x, mu, s2):
    return math.exp((x-mu)*(x-mu)/(2.0*s2)) / math.sqrt(2.0*math.pi*s2)


class ExpMax_1D:
    def __init__(self, x, k = 2):
        self.k = k
        self.m = len(x)
        self.x = x
        self.mu = [random.uniform(min(x), max(x)) for _ in range(k)]
        self.s2 = [1.0]*k
        self.fi = [1.0/k]*k
        self.w = [[0]*k for _ in range(m)]

    def e_step(self):
        for i in range(self.m):
            norm = 0.0
            for l in range(self.k):
                xi = self.x[i]
                norm += gauss(xi, self.mu[l], self.s2[l]) * self.fi[l]
            for j in range(self.k):
                self.w[i][j] = gauss(xi, self.mu[l], self.s2[l]) * self.fi[l]/norm

    def m_step(self):
        for j in range(self.k):
            norm = sum(self.w[:][j])
            self.fi[j] = norm / self.m
            mu_j = s2_j = 0
            for i in range(self.m):
                mu_j += self.w[i][j]
                s2_j += self.w[i][j] *(self.x[i] - self.mu[j])*(self.x[i] - self.mu[j])
            self.mu[j] = mu_j / norm
            self.s2[j] = s2_j / norm

    def fit(self, eps = 1e-4, iter_max = 1000):
        cur_L = self.loglike()
        for it in range(iter_max):
            prv_L = cur_L
            self.e_step()
            self.m_step()
            cur_L = self.loglike()
            if abs(cur_L-prv_L) < eps:
                break
        








X = np.random.multivariate_normal([0, 3], [[0.5, 0], [0, 0.8]], 20)
X = np.vstack((X, np.random.multivariate_normal([20, 10], np.identity(2), 50)))

gmm = GMM(X)
gmm.fit()

pass
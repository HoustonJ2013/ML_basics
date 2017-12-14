import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import nnls
class NMF:
    def __init__(self, k=10, maxiter = 10):
        self.k = k
        self.maxiter = maxiter
        self.V = None
        self.W = None
        self.H = None

    def init_matrix_(self):
        n,m = self.V.shape
        self.W = np.random.rand(n, self.k)
        self.H = np.random.rand(self.k, m)

    def fit(self, V):
        self.V = V
        self.init_matrix_()
        cost = []
        for iter in xrange(self.maxiter):
            self.H, r1 = lstsq(self.W, self.V)[0:2]
            self.H = np.clip(self.H, a_min=0, a_max= None)
            TEM, r2 = lstsq(self.H.T, self.V.T)[0:2]
            self.W = TEM.T
            self.W = np.clip(self.W, a_min=0, a_max= None)
            cost.append(np.sum((self.V - np.dot(self.W, self.H))**2))
        return self.W, self.H , cost

    # def fit2(self, V):
    #     self.V = V
    #     self.init_matrix_()
    #     cost = []
    #     for iter in xrange(self.maxiter):
    #         self.H, r1 = nnls(self.W, self.V)
    #         TEM, r2 = nnls(self.H.T, self.V.T)
    #         self.W = TEM.T
    #         cost.append(np.sum((self.V - np.dot(self.W, self.H))**2))
    #     return self.W, self.H , cost
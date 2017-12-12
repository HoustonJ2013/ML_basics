import numpy as np
from sklearn import metrics
class Kmeans:
    def __init__(self, K=3, max_iter=100, tol = 1e-4):
        self.X = None
        self.K = K
        self.centroids = None
        self.clusters = None
        self.max_iter_ = max_iter
        self.tol_ = tol

    def init_centroid_(self):
        if self.X is None:
            return "X is not initialized yet"
        m,n = self.X.shape
        choice_index = np.arange(m)
        choice_index_k = np.random.choice(choice_index, size=self.K, replace=False)
        #print(self.X[choice_index_k,:].shape)
        self.centroids = self.X[choice_index_k,:]

    def distance_(self,distance_fun = "square"):
        m, n = self.X.shape
        D = np.zeros((m, self.K))
        if distance_fun=="square":
            for i in xrange(self.K):
                D[:,i] = np.sum((self.X - self.centroids[i,:])**2, axis=1)
        return D

    def update_cluster(self):
        self.clusters = np.argmin(self.distance_(),axis=1)

    def update_centroids(self):
        centroid_old = np.copy(self.centroids)
        for i in xrange(self.K):
            self.centroids[i,:] = np.mean(self.X[self.clusters == i,:],axis = 0)
        dif = np.sum((centroid_old - self.centroids)**2)
        return dif

    def fit(self,X, K=3):
        self.K = K
        self.X = X
        self.init_centroid_()
        iter = 0
        while iter < self.max_iter_:
            self.update_cluster()
            dif = self.update_centroids()
           # print("Updating iteratio ", iter,dif)
            if (dif < self.tol_):
                break
            iter += 1
        return self.clusters

    def elbow_dist(self):
        dist = 0
        for i in xrange(self.K):
            cluster_mask = (self.clusters == i)
            dist = dist + np.sum((self.X[cluster_mask,:] - self.centroids[i,:]) ** 2)
        return dist

    def elbow_fit(self,X,K_max = 10):
        elbow_dist_list = []
        for k in xrange(1,K_max + 1):
            self.K = k
            self.fit(X, K=k)
            elbow_dist_list.append(self.elbow_dist())
        return elbow_dist_list

    def Silhouette_fit(self,X,K_max = 10):
        Silhouette_dist_list = []
        for k in xrange(2,K_max + 1):
            self.K = k
            clusters = self.fit(X, K=k)
            Silhouette_dist_list.append(metrics.silhouette_score(X, clusters, metric='euclidean'))
        return Silhouette_dist_list

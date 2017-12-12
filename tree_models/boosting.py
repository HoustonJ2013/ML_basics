import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt

class AdaBoostBinaryClassifier(object):
    '''
    INPUT:
    - n_estimator (int)
      * The number of estimators to use in boosting
      * Default: 50

    - learning_rate (float)
      * Determines how fast the error would shrink
      * Lower learning rate means more accurate decision boundary,
        but slower to converge
      * Default: 1
    '''

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1):

        self.base_estimator = DecisionTreeClassifier(max_depth=2)
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate

        # Will be filled-in in the fit() step
        self.estimators_ = []
        self.estimator_weight_ = np.zeros(self.n_estimator, dtype=np.float)

    def fit(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        Build the estimators for the AdaBoost estimator.
        '''

        n,m = x.shape
        weights = np.ones(y.shape, dtype=np.float) / n
        for i in range(self.n_estimator):
            estimator, weights, learning_rate = self._boost(x, y , weights)
            self.estimators_.append(estimator)
            self.estimator_weight_[i] = learning_rate
        pass

    def _boost(self, x, y, sample_weight):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels
        - sample_weight: numpy array

        OUTPUT:
        - estimator: DecisionTreeClassifier
        - sample_weight: numpy array (updated weights)
        - estimator_weight: float (weight of estimator)

        Go through one iteration of the AdaBoost algorithm. Build one estimator.

        You will need to do these steps:

        Fix the Decision Tree using the weights. You can do this like this: estimator.fit(X, y, sample_weight=sample_weight)
        Calculate the error term (estimator_error)
        Calculate the alphas (estimator_weight)
        Update the weights (sample_weight)
        '''

        estimator = clone(self.base_estimator)
        estimator.fit(x, y, sample_weight=sample_weight)
        y_predict = estimator.predict(x).reshape(y.shape)
        mask = y != y_predict  ## Calculate the wrong prediction
        estimator_error = np.sum(sample_weight[mask]) / np.sum(sample_weight)
        learning_rate = np.log((1 - estimator_error) / estimator_error)
        sample_weight[mask] = sample_weight[mask] * np.exp(learning_rate)
        return estimator, sample_weight, learning_rate

    def predict(self, x):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix

        OUTPUT:
        - labels: numpy array of predictions (0 or 1)
        '''
        y_pred = np.zeros((x.shape[0], ))
        for i, tree in enumerate(self.estimators_):
            tree_pred = tree.predict(x)
            tree_pred[tree_pred == 0] = -1
            y_pred += self.estimator_weight_[i] * tree_pred
        y_ = np.sign(y_pred)
        y_[y_==-1] = 0
        return y_


    def score(self, x, y):
        '''
        INPUT:
        - x: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - score: float (accuracy score between 0 and 1)
        '''

        y_pred = self.predict(x).reshape(y.shape)
        return np.sum(y_pred == y) / float(y.shape[0])

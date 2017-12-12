from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        '''
        tree_list = []
        sample_index = np.arange(num_samples)
        for i in xrange(num_trees):
            dt = DecisionTree(num_features=num_features)
            tree_index = np.random.choice(sample_index,num_samples,replace=True)
            X_tree = X[tree_index,:]
            Y_tree = y[tree_index]
            dt.fit(X_tree,Y_tree)
            tree_list.append(dt)
        return tree_list

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
        n_trees = len(self.forest)
        n_rows = X.shape[0]
        y_pred_list = []
        for j in xrange(n_rows):
            y_pred_ones = [self.forest[i].root.predict_one(X[j]) for i in xrange(n_trees)]
            y_pred_list.append(Counter(y_pred_ones).most_common(1)[0][0])
        return np.array(y_pred_list)

    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        y_pred = self.predict(X)
        n_right = len(y_pred[y_pred == y])
        n_total = len(y)
        return float(n_right)/n_total
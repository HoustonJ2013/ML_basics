import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode


class DecisionTree(object):
    '''
    A decision tree class.
    '''

    def __init__(self, impurity_criterion='entropy'):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None  # root Node
        self.feature_names = None  # string names of features (for interpreting
                                   # the tree)
        self.categorical = None  # Boolean array of whether variable is
                                 # categorical (or continuous)
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini

    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - feature_names: numpy array of strings
        OUTPUT: None

        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding
        label.
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or \
                                   isinstance(x, bool) or \
                                   isinstance(x, str)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - TreeNode

        Recursively build the decision tree. Return the root node.
        '''

        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return node

    def _entropy(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the entropy of the array y.
        '''

        ### YOUR CODE HERE
        n = float(len(y))
        Counter_ = Counter(y)
        H = 0
        for i, val in Counter_.iteritems():
            pc = val / n
            H -= pc*np.log2(pc)
            
            
        return H

    def _gini(self, y):
        '''
        INPUT:
            - y: 1d numpy array
        OUTPUT:
            - float

        Return the gini impurity of the array y.
        '''

        ### YOUR CODE HERE
        n = float(len(y))
        Counter_ = Counter(y)
        Gini = 0
        for i, val in Counter_.iteritems():
            pc = val / n
            Gini += pc**2
        return (1 - Gini)

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
            - split_index: int (index of feature)
            - split_value: int/float/bool/str (value of feature)
        OUTPUT:
            - X1: 2d numpy array (feature matrix for subset 1)
            - y1: 1d numpy array (labels for subset 1)
            - X2: 2d numpy array (feature matrix for subset 2)
            - y2: 1d numpy array (labels for subset 2)

        Return the two subsets of the dataset achieved by the given feature and
        value to split on.

        Call the method like this:
        >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

        X1, y1 is a subset of the data.
        X2, y2 is the other subset of the data.
        '''
        if isinstance(X[0,split_index], str):
            sindex_ = X[:,split_index] == split_value
        elif isinstance(X[0,split_index], bool):
            sindex_ = X[:,split_index] == split_value
        else:
            sindex_ = X[:,split_index] < split_value
        X1 = X[sindex_,:]
        y1 = y[sindex_]
        X2 = X[sindex_==False,:]
        y2 = y[sindex_==False]
        ### YOUR CODE HERE
        
        return X1, y1, X2, y2

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            - y: 1d numpy array
            - y1: 1d numpy array (labels for subset 1)
            - y2: 1d numpy array (labels for subset 2)
        OUTPUT:
            - float

        Return the information gain of making the given split.

        Use self.impurity_criterion(y) rather than calling _entropy or _gini
        directly.
        '''
        n = float(len(y))
        n1 = len(y1)
        n2 = len(y2)
        
        return self.impurity_criterion(y) - n1/n * self.impurity_criterion(y1) \
                - n2/n * self.impurity_criterion(y2)

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT:
            - index: int (index of feature)
            - value: int/float/bool/str (value of feature)
            - splits: (2d array, 1d array, 2d array, 1d array)

        Determine which feature and value to split on. Return the index and
        value of the optimal split along with the split of the dataset.

        Return None, None, None if there is no split which improves information
        gain.

        Call the method like this:
        >>> index, value, splits = self._choose_split_index(X, y)
        >>> X1, y1, X2, y2 = splits
        '''
        m,n = X.shape
        index = 0
        value = ""
        max_info_gain = 0
        for i in xrange(n):
            if isinstance(X[0,i], str):
                unique_str = np.unique(X[:,i])
                for str_c in unique_str:
                    X1, y1, X2, y2 = self._make_split(X,y,i,str_c)
                    _gain = self._information_gain(y, y1, y2)
                    if _gain > max_info_gain:
                        max_info_gain, index, value = _gain, i, str_c
            elif isinstance(X[0,i], bool):
                unique_bool = np.unique(X[:,i])
                for bool_c in unique_bool:
                    X1, y1, X2, y2 = self._make_split(X,y,i,bool_c)
                    _gain = self._information_gain(y, y1, y2)
                    if _gain > max_info_gain:
                        max_info_gain, index, value = _gain, i, bool_c
            else:
                percentiles_ = [25,50,75] # [10,20,30,40,50,60,70,80,90]
                for perc_ in percentiles_:
                    val_ = np.percentile(X[:,i],perc_)
                    X1, y1, X2, y2 = self._make_split(X,y,i,val_)
                    _gain = self._information_gain(y, y1, y2)
                    if _gain > max_info_gain:
                        max_info_gain, index, value = _gain, i, val_                    
            
        return index , value , self._make_split(X,y,index,value)

    def predict(self, X):
        '''
        INPUT:
            - X: 2d numpy array
        OUTPUT:
            - y: 1d numpy array

        Return an array of predictions for the feature matrix X.
        '''
        
        return np.array([self.root.predict_one(row) for row in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        y = y.reshape(y_pred.shape)
        #        print(y.shape,y_pred.shape,np.sum(y_pred == y))
        return np.sum(y_pred == y) / float(y.shape[0])

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)

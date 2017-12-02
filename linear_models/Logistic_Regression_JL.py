## Self constructed logistic regression
## The solver is gradient descent
import numpy as np

class GradientDescent(object):
    """Preform the gradient descent optimization algorithm for an arbitrary
    cost function.
    """
    def __init__(self, cost, gradient, predict_func,
                 fit_intercept=False,
                 alpha=0.01,
                 num_iterations=10000,
                 lamb=0):
        """Initialize the instance attributes of a GradientDescent object.

        Parameters
        ----------
        cost: The cost function to be minimized.
        gradient: The gradient of the cost function.
        predict_func: A function to make predictions after the optimizaiton has
            converged.
        alpha: The learning rate.
        num_iterations: Number of iterations to use in the descent.

        Returns
        -------
        self: The initialized GradientDescent object.
        """
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.lamb = lamb
        self.cost_hist = []
    def fit(self, X, y):
        """Run the gradient descent algorithm for num_iterations repititions.

        Parameters
        ----------
        X: A two dimenstional numpy array.  The training data for the
            optimization.
        y: A one dimenstional numpy array.  The training response for the
            optimization.

        Returns
        -------
        self:  The fit GradientDescent object.
        """
        if self.fit_intercept:
            X_new = self.X_add_ones(X)
            self.coeffs = np.ones((X_new.shape[1], 1))
            for i in range(self.num_iterations):
                grad = self.gradient(X_new,y,self.coeffs,lamb=self.lamb)
                if i%100 == 0:
                    self.cost_hist.append(self.cost(X_new,y,self.coeffs,lamb=self.lamb))
                self.coeffs = self.coeffs - (self.alpha * grad)
        else:
            self.coeffs = np.ones((X.shape[1], 1))
            for i in range(self.num_iterations):
                grad = self.gradient(X,y,self.coeffs,lamb=self.lamb)
                if i%100 == 0:
                    self.cost_hist.append(self.cost(X,y,self.coeffs,lamb=self.lamb))
                self.coeffs = self.coeffs - (self.alpha * grad)

    def predict(self, X):
        """Call self.predict_func to return predictions.

        Parameters
        ----------
        X: Data to make predictions on.

        Returns
        -------
        preds: A one dimensional numpy array of predictions.
        """
        return self.predict_func(X,self.coeffs)
    def X_add_ones(self,X):
        m,n = X.shape
        X_new = np.ones((m,n+1))
        X_new[:,0:n] = X
        return X_new

class Logistic_Regression_JL(object):

    def __init__(self,alpha = 0.001,fit_intercept=False,lamb = 0):
        self.coefs_ = None
        self.intercept = None
        self.X = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept ## Intercept
        self.lamb = lamb ## Regularization strength

    def sigmoid(self,X,coefs_):
        return 1.0 / (1 + np.exp(-np.dot(X,coefs_)))

    def cost_logistic(self,X,y,coefs_,lamb=0):
        n = y.shape[0]
        hx = self.sigmoid(X,coefs_)
        cost = -np.sum(y * np.log(hx) + (1 - y) * np.log( 1 - hx))
        ridge_cost = np.sum(coefs_*coefs_)
        return cost + lamb * ridge_cost

    def gradient(self,X,y,coefs_,lamb=0):
        hx = self.sigmoid(X,coefs_)
        y_reshape = y.reshape(y.shape[0], 1)
        ridge_grd = 2 * coefs_
        return np.dot((hx - y_reshape).T, X).T + lamb * ridge_grd

    def predict_proba(self,X,coefs_):
        if self.fit_intercept:
            X_new = self.X_add_ones(X)
            return self.sigmoid(X_new,coefs_)
        else:
            return self.sigmoid(X,coefs_)
    def predict(self,X,threshold = 0.5):
        predict = self.predict_proba(X,self.coefs_)
        predict[predict > threshold] = 1
        predict[predict <= threshold] = 0
        return predict
    def fit(self,X, y, alpha_=None):
        if alpha_ is None:
            alpha_ = self.alpha
        gd = GradientDescent(self.cost_logistic, self.gradient, self.predict, alpha=alpha_
                             ,fit_intercept=self.fit_intercept,lamb=self.lamb)
        gd.fit(X, y)
        self.coefs_ = gd.coeffs
    def score(self,X,y):
        y_pred = self.predict(X)
        y = y.reshape(y_pred.shape)
#        print(y.shape,y_pred.shape,np.sum(y_pred == y))
        return np.sum(y_pred == y) / float(y.shape[0])
    def X_add_ones(self,X):
        m,n = X.shape
        X_new = np.ones((m,n+1))
        X_new[:,0:n] = X
        return X_new
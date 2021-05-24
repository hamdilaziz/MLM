# -*- coding: utf-8 -*-
"""

@author: hamdi
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import random
from sklearn.ensemble import RandomForestRegressor

from abc import abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin,ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import is_classifier

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



__all__ = ["RFDClassifier",
           "RFSVMClassifier"]

def check_dissimilarity_matrix(diss_matrix, X):
    """
    Checks if the given dissimilarity matrix is valid:
    - the dissimilarity matrix must be a NxN matrix, N begin the number of instances in X (number of rows)
    - the diagonal elements of the dissimilarity matrix must be equal to 0

    :param diss_matrix: the dissimilarity matrix
    :param X: the dataset from which the dissimilarity matrix is supposed to be computed
    """
    M, N = diss_matrix.shape
    n_instances, n_features = X.shape

    if M != N:
        raise ValueError("The dissimilarity matrix must be squared")

    if M != n_instances:
        raise ValueError("The dissimilarity matrix must be a NxN matrix, N being the number of instances"
                         "in the given training set X")

    diag = np.sum(np.diagonal(diss_matrix))
    if diag != 0:
        raise ValueError("The diagonal elements of the dissimilarity matrix must all be equal to 0")

################## Base Random Forest Regrossor #########################################################

class BaseRFDRegressor(BaseEstimator, RegressorMixin):
    """
    Base class for Random Forest Dissimilarity (RFD) based learning
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 estimator: BaseEstimator,
                 forest: BaseForest = None):
        self.forest = forest
        self.estimator = estimator
        self.dissimilarity_matrix = None
        self.leaves_ = None

        if self.forest is None:
            self.forest = RandomForestRegressor(n_estimators=100)



    def fit(self, X, y):
        """
        :param X: an NxD input matrix, N being the number of training instances and D being the number of features
        :param y: an N-sized output vector that contains the true labels of the training instances
        :param dissim_matrix: an already built dissimilarity matrix or None
        :return: self
        """

        # step 1 - fit the forest on the training set
        try:
            check_is_fitted(self.forest, 'estimators_')
        except NotFittedError:
            self.forest.fit(X, y)

        # step 2 - compute the dissimilarity matrix from this forest
        self.dissimilarity_matrix = self._get_dissimilarity_matrix(X, y)
        check_dissimilarity_matrix(self.dissimilarity_matrix, X)

        # step 3 - fit the second estimator on the dissimilarity matrix
        self._fit_estimator(self.dissimilarity_matrix, y)

        return self

    @abstractmethod
    def _get_dissimilarity_matrix(self, X, y):
        """
        abstract method
        Compute the dissimilarity matrix from a given training set.

        This function is not supposed to be used alone. It is called in the fit function and
        suppose the self.forest to have been fitted on the same training set.
        The only reason to define a separate function here is to allow for overriding

        :param X: The training instances
        :param y: The true labels
        :return: The dissimilarity matrix computed from self.forest
        """
        pass

    def _fit_estimator(self, X_diss, y):
        """
        :param X_diss: a dissimilarity matrix, i.e. the dissimilarity representation of a training set
        :param y: the true labels
        :return: the fitted estimator
        """
        check_estimator(self.estimator)
        return self.estimator.fit(X_diss, y)

    def predict(self, X):
        """
        """
        check_is_fitted(self.estimator)
        X = check_array(X)
        X_diss = self.get_dissimilarity_representation(X)
        return self.estimator.predict(X_diss)

    @abstractmethod
    def get_dissimilarity_representation(self, X):
        """
        abstract method
        Compute the dissimilarity representation of X

        :param X:
        :return:
        """
        pass

################## Random Forest Regressor #############################################

class RFDRegressor(BaseRFDRegressor):
    """
    Random Forest Dissimilarity (RFD) based learning for Classification tasks
    """

    def __init__(self,
                 estimator: BaseEstimator = RandomForestRegressor(n_estimators=100),
                 forest: BaseForest = None):
        super().__init__(estimator, forest)

    def _get_dissimilarity_matrix(self, X, y):
        """
        Compute the dissimilarity matrix from a given training set.

        This function is not supposed to be used alone. It is called in the fit function and
        suppose the self.forest to have been fitted on the same training set.
        The only reason to define a separate function here is to allow for overriding

        :param X: The training instances
        :param y: The true labels
        :return: The dissimilarity matrix computed from self.forest
        """
        try:
            check_is_fitted(self.forest, 'estimators_')
        except NotFittedError:
            raise ValueError("This function must not be called alone. It is called by the fit function"
                             "to ensure that self.forest is fitted first on the same training set")

        # 1 - store all the leaves for all x in X
        # self.leaves_ is a NxL matrix, with N the number of instances and L the number of trees in the forest
        self.leaves_ = self.forest.apply(X)

        # 2 - compute the similarity matrix for the first tree
        # a is a N-sized vector with the leaves' number of the first tree for all x in X
        # sim is a NxN matrix with 0s and 1s
        a = self.leaves_[:, 0]
        sim = 1 * np.equal.outer(a, a)

        # 3 - for each tree in the forest update the similarity matrix by cumulating
        # the tree similarity matrices in sim
        for i in range(1, self.forest.n_estimators):
            a = self.leaves_[:, i]
            sim += 1 * np.equal.outer(a, a)

        # 4 - average and 1-sim to obtain the final dissimilarity matrix
        sim = sim / self.forest.n_estimators
        self.dissimilarity_matrix = 1 - sim
        return self.dissimilarity_matrix

    def get_dissimilarity_representation(self, X):
        """
        Compute the dissimilarity representation of X

        :param X: an instance (D-sized vector) or a dataset (NxD matrix)
        :return:
        """
        check_is_fitted(self.forest, 'estimators_')
        X_sim = np.empty((X.shape[0], 0))
        X_leaves = self.forest.apply(X)
        for xi in self.leaves_:
            matches = X_leaves == np.reshape(xi, (1, xi.shape[0]))
            sim = np.sum(matches, axis=1) / self.forest.n_estimators
            X_sim = np.append(X_sim, np.reshape(sim, (X.shape[0], 1)), axis=1)
        return 1 - X_sim
    
################################# base  Random Forest Classifier ###################################
class BaseRFDClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for Random Forest Dissimilarity (RFD) based learning
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 estimator: BaseEstimator,
                 forest: BaseForest = None):
        self.forest = forest
        self.estimator = estimator
        self.dissimilarity_matrix = None
        self.leaves_ = None

        if self.forest is None:
            self.forest = RandomForestClassifier(n_estimators=100)

        if not is_classifier(self.estimator):
            raise TypeError("The estimator must be a classifier")

    def fit(self, X, y):
        """
        :param X: an NxD input matrix, N being the number of training instances and D being the number of features
        :param y: an N-sized output vector that contains the true labels of the training instances
        :param dissim_matrix: an already built dissimilarity matrix or None
        :return: self
        """

        # step 1 - fit the forest on the training set
        try:
            check_is_fitted(self.forest, 'estimators_')
        except NotFittedError:
            self.forest.fit(X, y)

        # step 2 - compute the dissimilarity matrix from this forest
        self.dissimilarity_matrix = self._get_dissimilarity_matrix(X, y)
        check_dissimilarity_matrix(self.dissimilarity_matrix, X)

        # step 3 - fit the second estimator on the dissimilarity matrix
        self._fit_estimator(self.dissimilarity_matrix, y)

        return self

    @abstractmethod
    def _get_dissimilarity_matrix(self, X, y):
        """
        abstract method
        Compute the dissimilarity matrix from a given training set.

        This function is not supposed to be used alone. It is called in the fit function and
        suppose the self.forest to have been fitted on the same training set.
        The only reason to define a separate function here is to allow for overriding

        :param X: The training instances
        :param y: The true labels
        :return: The dissimilarity matrix computed from self.forest
        """
        pass

    def _fit_estimator(self, X_diss, y):
        """
        :param X_diss: a dissimilarity matrix, i.e. the dissimilarity representation of a training set
        :param y: the true labels
        :return: the fitted estimator
        """
        check_estimator(self.estimator)
        return self.estimator.fit(X_diss, y)

    def predict(self, X):
        """
        """
        check_is_fitted(self.estimator)
        X = check_array(X)
        X_diss = self.get_dissimilarity_representation(X)
        return self.estimator.predict(X_diss)

    @abstractmethod
    def get_dissimilarity_representation(self, X):
        """
        abstract method
        Compute the dissimilarity representation of X

        :param X:
        :return:
        """
        pass

####################### Random Forest Classifier #########################################
class RFDClassifier(BaseRFDClassifier):
    """
    Random Forest Dissimilarity (RFD) based learning for Classification tasks
    """

    def __init__(self,
                 estimator: BaseEstimator = RandomForestClassifier(n_estimators=100),
                 forest: BaseForest = None):
        super().__init__(estimator, forest)

    def _get_dissimilarity_matrix(self, X, y):
        """
        Compute the dissimilarity matrix from a given training set.

        This function is not supposed to be used alone. It is called in the fit function and
        suppose the self.forest to have been fitted on the same training set.
        The only reason to define a separate function here is to allow for overriding

        :param X: The training instances
        :param y: The true labels
        :return: The dissimilarity matrix computed from self.forest
        """
        try:
            check_is_fitted(self.forest, 'estimators_')
        except NotFittedError:
            raise ValueError("This function must not be called alone. It is called by the fit function"
                             "to ensure that self.forest is fitted first on the same training set")

        # 1 - store all the leaves for all x in X
        # self.leaves_ is a NxL matrix, with N the number of instances and L the number of trees in the forest
        self.leaves_ = self.forest.apply(X)

        # 2 - compute the similarity matrix for the first tree
        # a is a N-sized vector with the leaves' number of the first tree for all x in X
        # sim is a NxN matrix with 0s and 1s
        a = self.leaves_[:, 0]
        sim = 1 * np.equal.outer(a, a)

        # 3 - for each tree in the forest update the similarity matrix by cumulating
        # the tree similarity matrices in sim
        for i in range(1, self.forest.n_estimators):
            a = self.leaves_[:, i]
            sim += 1 * np.equal.outer(a, a)

        # 4 - average and 1-sim to obtain the final dissimilarity matrix
        sim = sim / self.forest.n_estimators
        self.dissimilarity_matrix = 1 - sim
        return self.dissimilarity_matrix

    def get_dissimilarity_representation(self, X):
        """
        Compute the dissimilarity representation of X

        :param X: an instance (D-sized vector) or a dataset (NxD matrix)
        :return:
        """
        check_is_fitted(self.forest, 'estimators_')
        X_sim = np.empty((X.shape[0], 0))
        X_leaves = self.forest.apply(X)
        for xi in self.leaves_:
            matches = X_leaves == np.reshape(xi, (1, xi.shape[0]))
            sim = np.sum(matches, axis=1) / self.forest.n_estimators
            X_sim = np.append(X_sim, np.reshape(sim, (X.shape[0], 1)), axis=1)
        return 1 - X_sim

######################### Minimal Learning Machine Regressor with Random Forest #########################
class RFDMLM_Regressor():
    def __init__(self):
        self.RF = RFDRegressor() # initialize the random forest regressor 
        
    def pinv_(self,D_x): 
        try:
            return np.linalg.inv(D_x.T @ D_x) @ D_x.T# if the matrice is inversible
        except Exception as e:
            return np.linalg.pinv(D_x)# compute the pseudo inverse 
    
    def fit(self,X,y):
        self.x_train = X
        self.y_train = y.reshape((len(y),1))# reshape to compute output matrix distances
        self.RF.fit(self.x_train,y) # fit the random forest 
        self.train_dissim = self.RF.get_dissimilarity_representation(self.x_train)# get RF dissimilarity matrix 
        self.D_y = cdist(self.y_train,self.y_train) # compute euclidean distances for output matrix
        self.B = self.pinv_(self.train_dissim) @ self.D_y# fit matrix of coefficients 
        
    def cost(self,y,x,d_x):
        y = np.array(y).reshape((1,len(y)))
        d_y  = cdist(y,self.y_train)# compute the vector of dictences for y
        return ((d_y**2 - (d_x@self.B)**2)**2)[0] 

    def optimse(self,x,d_x):
        J = lambda y: self.cost(y,x,d_x)# the fonction to minimize 
        out = least_squares(J,x0 =self.y_train.mean(axis=0),method='lm') #minimization of the cost function
        return out.x[0]

    def predict(self,X):
        self.test_dissim = self.RF.get_dissimilarity_representation(X)# get RF dissimlarité for X
        return np.array([self.optimse(x,d_x) for (x,d_x) in zip(X,self.test_dissim)])
    

####################### Minimal Learning Machine Classifier with Random Forest Regressor ###########################

def one_hot(y):
    # transform vector of classes with one hot encoding 
    y = [int(i) for i in y.tolist()]
    l = len(y)
    c = len(np.unique(y))
    y_oh = np.zeros((l, c))
    y_oh[np.arange(l), y] = 1
    return y_oh

class RFDMLM_Classifier():
    def __init__(self):

        self.RF = RFDClassifier()
        
    def pinv_(self,D_x):
        try:
            return np.linalg.inv(D_x.T @ D_x) @ D_x.T
        except Exception as e:
            return np.linalg.pinv(D_x)
    
    def fit(self,X,y):
        self.y_train = one_hot(y)
        self.x_train = X
        self.RF.fit(self.x_train,y)
        self.train_dissim = self.RF.get_dissimilarity_representation(self.x_train)
        self.D_y = cdist(self.y_train,self.y_train) 
        self.B = self.pinv_(self.train_dissim) @ self.D_y
        

    def predict(self, X, y=None):
        self.test_dissim = self.RF.get_dissimilarity_representation(X)
        # compute matrix of distances from input RPs
        # estimate matrix of distances from output RPs
        D_y_hat = self.test_dissim @ self.B
        return self.y_train[D_y_hat.argmin(axis=1),:].argmax(axis=1)
    
    def score(self,X,y):
        pred = self.predict(X)
        assert pred.shape == y.shape
        s = 0
        size = len(y)
        for i in range(size):
            if pred[i] == y[i]:
                s+=1
        return s/size
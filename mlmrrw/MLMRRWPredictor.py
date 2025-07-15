# method implementation
# maximizing variance reduction method
"""
(ML)Multi label classification
(SS)Semisupervised
(VR)Variance reduction method
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import pprint
import pandas as pd
import sys
from numpy.random import default_rng
import math
import threading
import random
from scipy.linalg import inv
from sklearn.metrics import pairwise_distances

import warnings




class MLMRRWPredictor(BaseEstimator, ClassifierMixin):
    """
    Will not implement as an extension of MLSSVRPRedictor due to changes on main fit algorithm. They will look alike, but we need to separate some common funcionalities.
    """
    
    def __init__(self, unlabeledIndex=None, tag = "" , hyper_params_dict = {}, XI = None ):
        super().__init__()
        self.hyper_params = hyper_params_dict
        self.unlabeledIndex = unlabeledIndex
        self.tag = tag
        self.XI = XI # the matrix that states if we want to believe on supervised or semisupervised. ONly the tr(XI) is filled with data
    
    def fit(self, X, y):
        y = y.copy()
        
        #rint(self.unlabeledIndex)
        #print(self.compatibilityMatrix)
        #print(self.unlabeledIndex)
        
        y.replace( {0:-1}, inplace=True )
        
        if not (self.unlabeledIndex is None):
            toSelect = y.index.intersection(self.unlabeledIndex)
            # print(f"Doing semi superv sim {X.shape}")
            
            y.loc[toSelect, :] = 0 # the algorithm states that 0 is unset/unlabeled, 1 is relevant and -1 is not relevant
            self.supervised_instances_amount = len(y.index) - len(self.unlabeledIndex)
        else:
            # print(f"Doing supervised sim")
            self.unlabeledIndex = pd.Index([0])
            toSelect = y.index.intersection(self.unlabeledIndex)
            y.loc[toSelect, :] = -1
            self.supervised_instances_amount = len(y.index) - len(self.unlabeledIndex)
            # just mimic with one the semisupervised setup
            # add unlabeled columns

        # if(noCompat):
        #    return self
        self.instances = X
        self.labels = y
        
        
        self.supervised_instances_amount = len(y.index) - len(self.unlabeledIndex)
        
        # print(self.labels)
        
        # gather on d_l the labeled samples and on d_u the unlabeled samples, take them to numpy +
        

        d_l = self.instances[  self.labels[self.labels.columns[0]] != 0 ].to_numpy()
        d_u = self.instances[  self.labels[self.labels.columns[0]] == 0 ].to_numpy()
        
        self.nx = np.append(d_l, d_u, axis = 0)
        
        y_l = self.labels[  self.labels[self.labels.columns[0]] != 0 ].to_numpy()
        v1 = np.ones(shape=[y_l.shape[0], 1] )
        
        y_u = self.labels[  self.labels[self.labels.columns[0]] == 0 ].to_numpy()
        v2 = np.ones(shape=[y_u.shape[0], 1] )
        
        v = np.append(  v1*self.hyper_params['XI_v1'] ,  v2*self.hyper_params['XI_v2'] , axis=0)
        psi_v = np.append( v1  , v2*0)
        
        ny = np.append(y_l, y_u, axis = 0)
        
        # print(self.instances)
        # print(d_l)
        # print(d_u)
        
        # print(y_l)
        # print(y_u)
        
        #print(nx)
        #print(ny)
        
        #print(v1)
        #print(v2)
        
        # gammas A and I are hyperparameters
        gamma = self.hyper_params['gamma']
        eps = self.hyper_params['eps']
        distances = pairwise_distances(self.nx)/(-2*gamma*gamma)
        K = np.exp(distances)
        I = np.eye( N = self.nx.shape[0] , M= self.nx.shape[0] )
        XI = np.eye( N = self.nx.shape[0] , M= self.nx.shape[0] )
        XI = XI*v
        self.XI = XI
        PSI = np.eye( N = self.nx.shape[0] , M= self.nx.shape[0] )
        PSI = PSI*psi_v
        n2 = self.nx.shape[0]*self.nx.shape[0]
        l = y_l.shape[0]
        # L = D -  W -> calculate this 
        
        H = np.sqrt( 2 - 2*K)
        B = np.where( (1-H) > eps, 0 ,1 )
        identity_negated = 1 - np.eye(N = B.shape[0] , M = B.shape[0])
        B = np.multiply( B , identity_negated  )
        W = np.multiply(H,B)
        degrees = W.sum(axis = 0)
        
        D = np.eye(N = W.shape[0] , M = W.shape[0])*degrees.T
        L = D - W
        # print(XI)
        # print(PSI)
        # print(K)
        #print(H)
        #print(B)
        #print(W)
        #print(degrees)
        #print(D)
        #print(L)
        r  = (PSI@K)@XI + (l*self.hyper_params['gamma_A']*I) + (((l*self.hyper_params['gamma_I']/n2)*L)@K)@XI
        #print( (PSI@K)@XI  )
        #print( (self.hyper_params['gamma_A']*I) )
        #print(r)
        self.theta_prime = inv(r)@ny
            
        #print("values")
        #print(theta_prime)
        return self

    def predict(self, X, print_prediction_log=False):
        
        pred, prob = self.predict_with_proba(X, print_prediction_log)
        
        return pred

    def predict_with_proba(self, X, print_prediction_log=False ,y_true = None):

        gamma = self.hyper_params['gamma']
        distances = pairwise_distances(X, self.nx)/(-2*gamma*gamma)
        K = np.exp(distances)
        F = (K@self.XI)@self.theta_prime
        
        pred = np.where(F >= 0 , 1, 0)
        prob = (F+1)/2
        return pred, prob


    def predict_proba(self, X, print_prediction_log=False):
        # check_is_fitted(self)
        pred, prob =  self.predict_with_proba(X,print_prediction_log)
        return prob

# -*- coding: utf-8 -*-
# linear_regression/regression.py
import numpy as np
import pandas as pd
import matplotlib as plt
import time
from tqdm import tqdm


def exeTime(func):
    """ Time-consuming calculation decorator """
    def newFunc(*args,**arg2):
        t0 = time.time()
        back = func(*args,**arg2)
        return back, time.time()-10
    return newFunc


def loadDataSet(filename):
    """ load data from data set
        Args:
            filename: file path
        Returns:
            X: training matrix
            Y：label matrix
    """
    numFea = len(open(filename).readline().split('\t')) - 1
    X = []
    Y = []
    
    file = open(filename)
    for line in tqdm(file.readlines()):
        lineArr = []
        curLine = line.strip().split('\t')
        
        for i in range(numFea):
            lineArr.append(float(curLine[i]))
        X.append(lineArr)
        Y.append(float(curLine[-1]))
        
    # test data analysis result
    # print(np.mat(x))
    # print(np.mat(y).T)
    return np.mat(X),np.mat(Y).T
# test function loadDataSet()
# x,y = loadDataSet('ex1data.txt')
    

def hypothesis(theta,X):
    """ hypthesis function -- hθ(x)=θ0 + θ1*x
        Args:
            theta: correlation coefficient matrix
            X：feature vector
        Returns:
            prediction:预测结果
    """
    predictions = (theta.T * X)[0,0]
    return predictions


def cost(theta, X, Y):
    """ cost function -- J(θ)= 1/2m * ∑((hθ(xi)-yi)^2)
    Args：
        theta: correlation coefficient matrix
        X: training matrix
        Y：label matrix
    Returns:
        errors: 预测误差
    """
    m = len(X)
    errors = (X*theta-Y).T * (X*theta-Y)/(2*m)
    return errors


@exeTime
def bgd(rate, maxloop, epsilon, X, Y):
    """ batch gredient descent
        Args:
            rate: learning rate
            maxloop: maximum iteration
            epsilon: convergent accuracy
            X: training matrix
            Y: label matrix
        Returns:
            (theta,errors,thetas)、timeConsumed
    """
    # m:num of training examples ; n = num of features
    m,n = X.shape
    # initilize vector theta
    theta = np.zeros((n,1))
    # iterations
    count = 0
    # converge flag
    converged = False
    # learning error
    error = float('inf')
    # total learning error
    errors = []
    # all of the theta
    thetas = {}
    
    for j in tqdm(range(n)):
        thetas[j] = [theta[j,0]]
        
    # start iterative
    while count<=maxloop:
        if(converged):
            break
        count = count + 1
        
        # simultaneous updata theta j
        for j in range(n):
            # dJ(θ)/dθ = (∑(θ0+θ1*xi - yi)*xi)/m 
            derivative = (Y-X*theta).T * X[:,j]/m
            # θi := θi - α*(dJ(θi)/dθ)
            theta[j,0] = theta[j,0] + rate*derivative
            thetas[j].append(theta[j,0])
            
        # calculate cost after one iteration
        error = cost(theta,X,Y)
        errors.append(error[0,0])
        
        # judge converged
        if(error < epsilon):
            converge = True
    
    return theta,errors,thetas


def standardize(X):
    """ features standardize
        Args:
            X: training matrix
        Returns:
            standard X
    """
    m,n = X.shape
    for j in range(n):
        # features of each hypotheses
        features = X[:,j]
        # mean value
        meanVal = features.mean(axis=0)
        # Standard Deviation 
        std = features.std(axis=0)
        
        if std !=0:
            X[:,j] = (features - meanVal)/std
        else:
            X[:,j] = 0
    return X


def normalize(X):
    """ features normalize
        Args:
            X: training matrix
        Returns:
            normally X
    """
    m,n = X.shape
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        
        if diff !=0:
            X[:,j] = (features - minVal)/diff
        else:
            X[:,j] = 0
    return X
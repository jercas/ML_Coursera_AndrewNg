# -*- coding: utf-8 -*-
"""
Created on SUN Dec 10 15:53:30 2017

@author: jercas

v 1.0 First time coding end on SUN Dec 10 16:00:40 2017
"""

import neuralNetwork as nn
import numpy as np
from sklearn import datasets
from scipy.io import loadmat

data = loadmat('handwritten_digits.mat')
weights = loadmat('weights.mat')
weights = [weights['Theta1'], weights['Theta2']]

X = np.mat(data['X'])
y = np.mat(data['y'])

result = nn.train(X, y, hiddenLayers=1 ,hiddenNum=25, weights=weights, precision=0.5)
print("Run {0} iterations".format(result['iters']))
print("Error is : {0}".format(result['error']))
print("Weights is : {0}".format(result['weights']))
print("Success is: {0}".format(result['success']))
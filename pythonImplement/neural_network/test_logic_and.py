# -*- coding: utf-8 -*-
"""
Created on SUN Dec 10 14:28:40 2017

@author: jercas

v 1.0 First time coding end on SUN Dec 10 15:53:40 2017
"""

import neuralNetwork as nn
import numpy as np

data = np.mat([
	[0,0,0],
	[1,0,0],
	[0,1,0],
	[1,1,1]
])

X = data[:, 0:2]
y = data[:, 2]

result = nn.train(X, y, hiddenNum=0, alpha=10, maxIters=5000, precision=0.01)

print("Run {0} iterations".format(result['iters']))
print("Error is : {0}".format(result['error']))
print("Weights is : {0}".format(result['weights'][0]))
print("Success is: {0}".format(result['success']))
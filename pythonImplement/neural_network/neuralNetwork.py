# -*- coding: utf-8 -*-
"""
Created on THU Dec 7 14:51:00 2017

@author: jercas

v 1.0 First time coding end on FRI Dec 8 20:00:00 2017
	  complete expect test(logistic classification & multiple classification), matching matlab version
"""
import numpy as np

"""
	Neural Network default settings:
		- automatic random initialize weights matrices
		- don't include hidden layer
		- hidden layers' units = 5
		- learning rate = 1.0
		- don't regularization
		- error precision = 0.01
		- max iteration = 50
"""


def sigmoid(z):
	"""
		sigmoid(z)

		sigmoid function: g(z)=1 / (1 + e^(-z))

		Parameters
		-----------
			z:input unactivated value

		Returns
		--------
			result: processed result by sigmoid function
	"""
	result = 1.0 / (1.0 + np.exp(-z))
	return result


def sigmoidDerivative(a):
	"""
		sigmoidDerivative(z)

		sigmoid function's derivative

		Parameters
		----------
			a: input activated value

		Returns
		-------
			result: derivative of sigmoid function result
	"""
	result = np.multiply(a, (1.0-a))
	return result

def randInitializeWeights(inputs, hiddenNum, hiddenLayers, outputs, epsilon):
	"""
		randInitializeWeights(input, hidden, units, output, epsilon)

		Random initialize the weights/theta/θ matrix

		Parameters
		----------
			inputs: number of units in input layer
			hiddenNum: number of units in each hidden layer
			hiddenLayers: number of hidden layers
			outputs: number of units in output layer / class number
			epsilon: bias variables

		Returns
		-------
			weights: a weights/theta/θ matrix
	"""
	# units num of each hidden layer
	hiddens = [hiddenNum for i in range(hiddenLayers)]
	# the list of units num in each layer
	unitsOfLayer = [inputs] + hiddens + [outputs]
	# all of the theta/θ in neural network
	weights = []

	for index, layer in enumerate(unitsOfLayer):
		# l-1 theta in l layers neural network
		if index == len(unitsOfLayer) - 1:
			break

		nextLayer = unitsOfLayer[index + 1]
		# random initialize with bias
		weight = np.random.rand(nextLayer, layer+1) * 2 * epsilon - epsilon
		weights.append(weight)

	return  weights


def unroll(matrices):
	"""
	unroll(matrices)

	unroll input matrices into vector for learning algorithm

	Parameters
	----------
		matrices: weights/theta/θ matrices
	Returns
	-------
		vector: unrolled matrix
	"""
	vector = []
	for matrix in matrices:
		# unroll input matrix into vector
		vec = matrix.reshape(1,-1)[0]
		vector = np.concatenate((vec,vector))
	return vector


def roll(vector, shapes):
	"""
		roll(vector)

		restore each of unrolled matrices from vector

		Parameters
		----------
			vector: vector format weights/theta/θ use to easily calculate gradient
			shapes: a list (contain array-like elements) store all of the weights matrices' shape in order to restore them

		Returns
		-------
			matrices: restored matrix
	"""
	matrices = []
	beginIdx = 0
	for shape in shapes:
		# get matrix's size
		endInd = beginIdx + shape[0] * shape[1]
		matrix = vector[beginIdx:endInd].reshape(shape)
		beginIdx = endInd
		matrices.append(matrix)
	return matrices


def cost(weights, y, lambdaValue, X=None, a=None):
	"""
		cost(weights, y, lambdaValue, X=None. a=None)

		computer cost of prediction of trained model

		Parameters
		----------
			weights: weights vectors
			y: labels vector
			lambdaValue: bias value for regularization
			X: input training data
			a: output activations by each neural units in every layer(l>1)

		Returns
		-------
			cost: calculated error of prediction
	"""
	m = y.shape[0]
	if a is None:
		# No activations of each layer, first run feedforword propagation get all of the activations
		a = fp(weights, X)

	# cost function without regularization
	error = -np.sum(np.multiply(y.T, np.log(a[-1])) + np.multiply((1-y).T, np.log(1- a[-1])))

	# regularize process
	reg = -np.sum([ np.sum(weight[:, 1:])  for weight in weights ])

	costValue = (1.0/m) * error + (1.0/(2*m)) * lambdaValue * np.square(reg)
	return costValue


def fp(weights, X):
	"""
		fp(weights, X)

		feedforword propagation algorithm

		Parameters
		----------
			weights: weights vector
			X: input training data

		Returns
		-------
			a: output activations by each neural units in every layer(l>1)
	"""
	# layer num in the neural network
	layers = len(weights) + 1
	# activation list
	a = [i for i in range(layers)]

	# calculate each layer's output
	for l in range(layers):
		if l == 0:
			a[l] = X.T
		else:
			# activated output
			z_l = weights[l-1] * a[l-1]
			a[l] = sigmoid(z_l)
		# add biad except output layer
		if l != layers-1:
			a[l] = np.concatenate((np.ones((1, a[l].shape[1])), a[l]))
	return  a


def bp(weights, a, y, lambdaValue):
	"""
		bp(weights, a, y, lambdaValue)

		backforword propagation algorithm

		Parameters
		----------
			weights: weights/theta/θ  vector
			a: output activations by each neural units in every layer(l>1)
			y: labels  vector
			lambdaValue: bias for regularization

		Returns
		-------
			D: updated gradient weights
	"""
	m = y.shape[0]
	# layer num in the neural network
	layers = len(weights) + 1
	# error of each unit in different layer(except input layer)
	d = [ i for i in range(layers) ]
	# accumulator to add up values as go along compute all of partial derivative
	delta = [ np.zeros(weight.shape) for weight in weights ]

	# calculate each units' delta
	# [::-1] back forword
	for l in range(layers)[::-1]:
		if l == 0:
			# no error in input layer
			break
		if l == layers - 1:
			# output layer error
			d[l] = a[l] - y.T
		else:
			# hidden layer error without regularization
			d[l] = np.multiply((weights[l][:, 1:].T * d[l+1]), sigmoidDerivative(a[l][1:, :]))

	# calculate gradient of weights
	for l in range(layers)[0:layers - 1]:
		delta[l] = delta[l] + d[l+1] * a[l].T

	# calculate update value of weights
	D = [ np.zeros(weight.shape) for weight in weights ]
	for l in range(len(weights)):
		weight = weights[l]
		# bias term update
		D[l][:, 0] = (1.0/m)  * (delta[l][:, 0].reshape(1, -1))
		# weight term update
		D[l][:, 1:] = (1.0/m) * (delta[l][:, 1:] + lambdaValue*weight[:, 1:])
	return D


def updateWeights(m, weights, D, alpha):
	"""
		updateWeights(m, weights, D, alpha)

		update weights use the learning experience from predict error get from bp

		Parameters
		----------
			m: labels num
			weights: weight/theta/θ
			D: gradient weight
			alpha: learning rate

		Returns
		-------

	"""
	for l in range(len(weights)):
		weights[l] = weights[l] - alpha*D[l]
	return  weights


def gradientDescent(weights, X, y, alpha, lambdaValue):
	"""
		gradientDescent(weights, X, y, alpha, lambdaValue)

		gradient descent algorithm

		Parameters
		----------
	        weights: weight/theta/θ
			X: input training data
			y: labels vector
			alpha: learning rate
			lambdaValue: bias for regularization

		Returns
		-------
			J: trained cost error after one epoch
			weights: trained weight/theta/θ after one epoch
	"""
	# examples num, features num
	m,n = X.shape
	# feedforword propagation calculate activations of each neural unit
	a = fp(weights, X)
	# calculate cost of prediction
	J = cost(weights, y, lambdaValue, a=a)
	# backforword propagation calculate update gradient weights
	D = bp(weights, a, y, lambdaValue)
	# update weights
	weights = updateWeights(m, weights, D, alpha)

	# judge gradient explode
	if np.isnan(J):
		J = np.inf
	return  J, weights


def gradientCheck(weights, X, y, lambdaValue):
	"""
		gradientCheck(weights, X, y, lambdaValue)

		gradient check to estimate does gradient descent real work right

		Parameters
		----------
			weights: weight/theta/θ
			X: input training data
	        y: labels vector
			lambdaValue: bias for regularization

		Returns
		-------
			True: gradient descent is work as expect
			False: gradient descent isn't work as expect
	"""
	# examples num, features num
	m,n = X.shape
	# feedforword propagation calculate activations of each neural unit
	a = fp(weights, X)
	# calculate cost of prediction
	J = cost(weights, y, lambdaValue, a=a)
	# backforword propagation calculate update gradient weights
	D = bp(weights, a, y, lambdaValue)

	DVec = unroll(D)
	weightVec = unroll(weights)

	# gradient approximate range
	epsilon = 1e-4
	# numerically approximating derivative of weights
	gradApprox = np.zeros(DVec.shape)
	# get matrices's shape for roll it into matrix from vector
	shapes = [ weight.shape for weight in weights ]

	for index,item in enumerate(weightVec):
		# numerically calculate gradient
		weightVec[index] = item - epsilon
		Jminus = cost(roll(weightVec, shapes), y ,lambdaValue, X=X)
		weightVec[index] = item + epsilon
		Jplus  = cost(roll(weightVec, shapes), y ,lambdaValue, X=X)
		gradApprox[index] = (Jplus - Jminus) / (2*epsilon)

	# degree of approximation
	diff = np.linalg.norm(gradApprox - DVec)
	print(diff)
	if diff < 1e-2:
		return  True
	else:
		return  False


def adjustLabels(y):
	"""
		adjustLabels(y)

		adjust label for right classificate

		Parameters
		----------
			y: lables vector

		Returns
		-------
			y: adjusted labels
	"""
	# adjust if labels are logistic mark
	if y.shape[1] == 1:
		# like--3,4,5
		# np.ravel() -- Dimensionality reduction, turn n-d matrix to 1-d array
		#               Different from np.flatten which return a copy of origin, but return a view of origin
		classes = set(np.ravel(y))
		classNum = len(classes)
		# implement array in python, index start as 0
		minClass = min(classes)

		# multiple classification
		if classNum>2:
			yAdjusted = np.zeros((y.shape[0], classNum), np.float64)
			for row, label in enumerate(y):
				yAdjusted[row, label-minClass] = 1
		# binary classification
		else:
			yAdjusted = np.zeros((y.shape[0], 1), np.float64)
			for row, label in enumerate(y):
				if label!=minClass:
					yAdjusted[row, 0] = 1.0
		return yAdjusted
	# already logistic label
	# like--[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]
	return  y


def train(X, y, weights=None, hiddenLayers=0, hiddenNum=5, epsilon=1, alpha=1, lambdaValue=0, precision=0.01, maxIters=50):
	"""
		train(X, y, weights=None, )

		neural network training

		Parameters
		----------
			X: input training data
			y: labels vector
			weights:
			hiddenNum: number of units in each hidden layer
			hiddenLayers: number of hidden layers
			epsilon: bias variables
			alpha: learning rate
			lambdaValue: bias value for regularization
			precision: error precision
			maxIters: maximum iterations

		Returns
		-------
			error: predict error
			weights: trained weights
			iters: training iteration num
			success: accuracy
	"""
	# training examples, features
	m, n = X.shape
	# adjust labels
	y = adjustLabels(y)
	classNum = y.shape[1]

	# initialize weights
	if weights is None:
		weights = randInitializeWeights(
			inputs=n,
			hiddenNum=hiddenNum,
			hiddenLayers=hiddenLayers,
			outputs=classNum,
			epsilon=epsilon
		)

	# gradient check
	print("-----Doing Gradient Checking-----")
	checked = gradientCheck(weights, X, y, lambdaValue)
	# gradient descent works well
	if checked:
		for i in range(maxIters):
			error, weights = gradientDescent(weights, X, y, alpha=alpha, lambdaValue=lambdaValue)
			# satify predict accuracy, end train
			if error<precision:
				break
			# gradient explode, end train
			if error == np.inf:
				break
		# after training, judge predicate result precision
		if	error < precision:
			success = True
		else:
			success = False
		return {
			'error': error,
			'weights': weights,
			'iters': i,
			'success': success
		}
	else:
		print("-----Error: Gradient Checking Failed!!!-----")
		return{
			'error': None,
			'weights': None,
			'iters': 0,
			'success': False
		}


def predict(X, weights):
	"""
		predict(X, weights)

		test train result

		Parameters
		----------
			X: test data set
			weights: trained weights/theta/θ

		Returns
		-------
			a: activations
	"""
	a = fp(weights, X)
	return a[-1]
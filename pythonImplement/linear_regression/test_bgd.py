# -*- coding: utf-8 -*-
# linear_regression/test_bgd.py
import regression
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

if __name__ == "__main__":
    X,Y = regression.loadDataSet('./ex1data.txt')
    m,n = X.shape
    # add all θ0 in each hypothesis equals 1
    X = np.concatenate((np.ones((m,1)),X),axis=1)

    # parameters setting
    rate = 0.01
    maxloop = 1500
    epsilon = 0.01
    
    # get learning result
    result, timeConsumed = regression.bgd(X, Y, rate, maxloop, epsilon)
    theta, errors, thetas = result
    
    # plot fitting line
    fittingFig = plt.figure()
    ax = fittingFig.add_subplot(111)
    
    # plot scatters
    trainingSet = ax.scatter(X[:,1].flatten().A[0],Y[:,0].flatten().A[0])
    # plot line
    xCopy = X.copy()
    xCopy.sort(0)
    yHat = xCopy*theta
    fittingLine, = ax.plot(xCopy[:,1],yHat,color='g')
    
    ax.set_title('bgd: rate={0}, maxloop={1}, epsilon={2}, time={3}'.format(rate,
                       maxloop,epsilon,timeConsumed))
    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel('Profit in $10,000s')
    plt.legend([trainingSet, fittingLine], ['Trainging Set', 'Linear Regression'])
    plt.show()
    
    # plot errors line
    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
    # plot line
    errorsline = ax.plot(range(len(errors)),errors,color='r')
    
    ax.set_title('converge % gradient descent judgement')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')
    plt.legend(errorsline,'errors')
    plt.show()
    
    # plot gradient descent coutour graph
    size = 100
    theta0Vals = np.linspace(-10,10,size)
    theta1Vals = np.linspace(-2,4,size)
    JVals = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            col = np.matrix([ [theta0Vals[i]], [theta1Vals[j]] ])
            JVals[i,j] = regression.cost(col,X,Y)
            
    theta0Vals, theta1Vals = np.meshgrid(theta0Vals, theta1Vals)
    JVals = JVals.T
    
    # plot three-dimensional system of coordinate
    contourSurf = plt.figure()
    ax = contourSurf.gca(projection='3d')
    ax.plot_surface(theta0Vals, theta1Vals, JVals, rstride=2, cstride=2, 
                    alpha=0.3, cmap=cm.rainbow, linewidth=0, antialiased=False)
    ax.plot(thetas[0],thetas[1],'rx')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')
    plt.show()
    
    # plot contour 
    contourFig = plt.figure()
    ax = contourFig.add_subplot(111)
    cs = ax.contour(theta0Vals, theta1Vals, JVals, np.logspace(-2,3,20))
    
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    plt.clabel(cs, inline=1, fontsize=10)
    
    # plot optimum
    ax.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)
    
    # plot gradient descent
    ax.plot(thetas[0], thetas[1], 'rx', markersize=3, linewidth=1)
    ax.plot(thetas[0], thetas[1], 'r-')
    
    plt.show()
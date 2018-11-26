"""
Create:25 November,2018
Author:Lekun ZHUANG
Github:https://github.com/LekunZHUANG
"""
from numpy import *
import matplotlib.pyplot as plt

#get the true function
def getTrueFunction(xo = linspace(0, 0.98, 50).reshape(50, 1)):
    yo = sin(xo**2+1)
    #plt.plot(xo, yo, color='blue')
    return xo, yo

#generate the data sample of the training set
def generateTrainingData():
    x = random.uniform(0, 1, 50).reshape(50, 1)
    e = random.normal(0, 0.02, 50).reshape(50, 1)
    y = sin(x**2+1) + e
    #plt.scatter(x, y, color='red')          #train set plot
    return x, y

#solve the beta via LS
def solveBeta(m):
    x, y = generateTrainingData()
    X = ones(50).reshape(50, 1)
    for i in range(1, m+1):
        X = hstack((X, x**i))
    b = linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
    #print(b)
    return b

#get the test function via beta which was trained by the train set
def getTestFunction(m , xt = linspace(0, 0.98, 50)):
    #xt = linspace(0, 0.98, 50)              #xt = [0:0.02:1]
    b = solveBeta(m)
    yt = 0
    for i in range(0, m+1):
        yt = yt + b[i]*(xt**i)
    #plt.plot(xt, yt, color='green')              #test function plot
    return xt, yt

#compute with the training
def computeTraining(M):
    MSE_Training = []
    for m in range(1, M+1):
        mse = 0
        for j in range(200):
            x, y = generateTrainingData()
            yo = getTrueFunction(x)[1]
            diffsquare = (y - yo)**2
            Eaverage = diffsquare.sum() / 50
            mse = mse + Eaverage
        MSE_Training.append(mse/200)
    return MSE_Training
    # M = [1, 2, 3, 4, 5, 6, 7, 8]
    # plt.plot(M, MSE_Training, color='red')

def computeTest(M):
    MSE_Test = []
    for m in range(1, M+1):
        mse = 0
        for j in range(200):
            xt, yt =getTestFunction(m)
            yo = getTrueFunction(xt)[1]
            diffsquare = (yt - yo)**2
            Eaverage = diffsquare.sum()/50
            mse = mse + Eaverage
        MSE_Test.append(mse/200)
    return MSE_Test

def PolyRegression():
    fig = plt.figure("PolyRegression")
    for i in range(1, 9):
        fig.add_subplot(4, 2, i)
        xo, yo = getTrueFunction()
        x, y = generateTrainingData()
        xt, yt =getTestFunction(i)
        plt.plot(xo, yo, color='blue', label='true function')
        plt.scatter(x, y, color='red', label='training set')
        plt.plot(xt, yt, color='green', label='test function')
    plt.legend(loc=0, )

def meanSquareError():
    plt.figure("MeanSquareError")
    M = 8
    Mspace = linspace(1, M, M)
    MSE_Training = computeTraining(M)
    MSE_Test = computeTest(M)
    MSE = []
    for i in range(M):
        MSE.append((MSE_Training[i] + MSE_Test[i]) / 2)
    plt.plot(Mspace, MSE_Training, color='red', label='MSE_Training')
    plt.plot(Mspace, MSE_Test, color='green', label='MSE_Test')
    plt.plot(Mspace, MSE, color='blue', label='MSE')
    plt.legend(loc=0, )

def computeBiasVariance(m,xe):
    Bias = []
    Variance = []
    for k in range(1, m+1):
        xt, yt = getTestFunction(m)
        Eyt = yt.mean()
        ye = getTrueFunction(xe)[1]
        bias = (Eyt - ye) ** 2
        Bias.append(bias)
        variance = 0
        for i in range(len(xt)):
            variance = variance + (yt[i] - Eyt) ** 2
        variance = variance / len(xt)
        Variance.append(variance)
    Bias = array(Bias)
    Variance = array(Variance)
    #print(Bias)
    #print(Variance)
    return Bias, Variance

def BiasVariance(xe):
    plt.figure("The relation of m, bias, variance:" + str(xe))
    m = 8
    Mspace = linspace(1, m, m)
    # xe = linspace(0, 0.9, 10)
    # Bias = zeros(8)
    # Variance = zeros(8)
    # for j in range(10):
    #     bias, variance = computeBiasVariance(m, xe[j])
    #     Bias = Bias + bias
    #     Variance = Variance = variance
    # Bias = Bias/8
    # Variance = Variance/8
    Bias, Variance = computeBiasVariance(m, xe)
    GeneError = Bias + Variance
    plt.plot(Mspace, Bias, color='red', label='Bias')
    plt.plot(Mspace, Variance, color='green', label='Variance')
    plt.plot(Mspace, GeneError, color ='blue', label='Generalization error')
    plt.legend(loc=0, )



PolyRegression()
meanSquareError()
BiasVariance(0.2)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Generate dataset
def generateData():
    np.random.seed(0)
    x = np.linspace(-5, 5, 1000)
    y = 2 * x - 3 + np.random.normal(0, 5, 1000)
    return x, y

# Shuffle data and split into train and test sets
def trainTestSplit(x, y, splitRatio=0.8):
    indices = np.random.permutation(len(x))
    trainSize = int(len(x) * splitRatio)
    trainIdx, testIdx = indices[:trainSize], indices[trainSize:]
    xTrain, xTest = x[trainIdx], x[testIdx]
    yTrain, yTest = y[trainIdx], y[testIdx]
    return xTrain, xTest, yTrain, yTest

# Stochastic Gradient Descent with shuffling
def stochasticGradientDescentShuffle(x, y, xTest, yTest, learningRate=0.001, epochs=1000):
    def updateParams(params, sample):
        x_i, y_i = sample
        y_pred = params[0] + params[1] * x_i
        error = y_pred - y_i
        newB0 = params[0] - 2*learningRate * error
        newB1 = params[1] - 2*learningRate * error * x_i
        return (newB0, newB1)

    def computeError(params, x, y):
        return np.mean((params[0] + params[1] * x - y)**2)

    params = (20, 20)  # initial intercept and slope
    errorsTrain, errorsTest = [], []
    for _ in range(epochs):
        shuffledData = np.random.permutation(list(zip(x, y)))
        for sample in shuffledData:
            params = updateParams(params, sample)
        errorsTrain.append(computeError(params, x, y))
        errorsTest.append(computeError(params, xTest, yTest))
    return params[0], params[1], errorsTrain, errorsTest

# Stochastic Gradient Descent without shuffling
def stochasticGradientDescentNoShuffle(x, y, xTest, yTest, learningRate=0.001, epochs=1000):
    def updateParams(params, sample):
        x_i, y_i = sample
        y_pred = params[0] + params[1] * x_i
        error = y_pred - y_i
        newB0 = params[0] - 2*learningRate * error
        newB1 = params[1] - 2*learningRate * error * x_i
        return (newB0, newB1)

    def computeError(params, x, y):
        return np.mean((params[0] + params[1] * x - y)**2)

    params = (20, 20)  # initial intercept and slope
    errorsTrain, errorsTest = [], []
    for _ in range(epochs):
        for i in range(len(x)):
            params = updateParams(params, (x[i], y[i]))
        errorsTrain.append(computeError(params, x, y))
        errorsTest.append(computeError(params, xTest, yTest))
    return params[0], params[1], errorsTrain, errorsTest

# Gradient Descent on mini batch
def miniBatchGradientDescent(x, y, xTest, yTest, batchSize=50, learningRate=0.01, epochs=100):
    def updateParams(params, batch):
        xBatch, yBatch = zip(*batch)
        yPred = params[0] + params[1] * np.array(xBatch)
        error = yPred - np.array(yBatch)
        newB0 = params[0] - 2*learningRate * np.mean(error)
        newB1 = params[1] - 2*learningRate * np.mean(error * np.array(xBatch))
        return (newB0, newB1)

    def computeError(params, x, y):
        return np.mean((params[0] + params[1] * x - y)**2)

    params = (0, 0)  # initial intercept and slope
    errorsTrain, errorsTest = [], []
    for _ in range(epochs):
        shuffledData = np.random.permutation(list(zip(x, y)))
        for i in range(0, len(shuffledData), batchSize):
            batch = shuffledData[i:i+batchSize]
            params = updateParams(params, batch)
        errorsTrain.append(computeError(params, x, y))
        errorsTest.append(computeError(params, xTest, yTest))
    return params[0], params[1], errorsTrain, errorsTest

# Plot error graphs
def plotErrors(errorsTrain, errorsTest, title):
    plt.plot(range(len(errorsTrain)), errorsTrain, label='Train')
    plt.plot(range(len(errorsTest)), errorsTest, label='Test')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

# Print final errors using tabulate
def printFinalErrors(errorsTrainSGDShuffle, errorsTestSGDShuffle, errorsTrainSGDNoShuffle, errorsTestSGDNoShuffle, errorsTrainMiniBatch, errorsTestMiniBatch):
    table = [["Stochastic GD with Shuffling", errorsTrainSGDShuffle[-1], errorsTestSGDShuffle[-1]],
             ["Stochastic GD without Shuffling", errorsTrainSGDNoShuffle[-1], errorsTestSGDNoShuffle[-1]],
             ["Mini Batch GD", errorsTrainMiniBatch[-1], errorsTestMiniBatch[-1]]]
    headers = ["Method", "Train Error", "Test Error"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

# Generate data
x, y = generateData()

# Split data into train and test sets
xTrain, xTest, yTrain, yTest = trainTestSplit(x, y)

# Stochastic Gradient Descent with shuffling
b0SGDShuffle, b1SGDShuffle, errorsTrainSGDShuffle, errorsTestSGDShuffle = stochasticGradientDescentShuffle(xTrain, yTrain, xTest, yTest)
plotErrors(errorsTrainSGDShuffle, errorsTestSGDShuffle, 'Stochastic GD with Shuffling')

# Stochastic Gradient Descent without shuffling
b0SGDNoShuffle, b1SGDNoShuffle, errorsTrainSGDNoShuffle, errorsTestSGDNoShuffle = stochasticGradientDescentNoShuffle(xTrain, yTrain, xTest, yTest)
plotErrors(errorsTrainSGDNoShuffle, errorsTestSGDNoShuffle, 'Stochastic GD without Shuffling')

# Gradient Descent on mini batch
b0MiniBatch, b1MiniBatch, errorsTrainMiniBatch, errorsTestMiniBatch = miniBatchGradientDescent(xTrain, yTrain, xTest, yTest)
plotErrors(errorsTrainMiniBatch, errorsTestMiniBatch, 'Mini Batch GD')

# Print final errors
print("Final errors:")
printFinalErrors(errorsTrainSGDShuffle, errorsTestSGDShuffle, errorsTrainSGDNoShuffle, errorsTestSGDNoShuffle, errorsTrainMiniBatch, errorsTestMiniBatch)


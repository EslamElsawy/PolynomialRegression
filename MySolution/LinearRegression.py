__author__ = 'eslamelsawy'

import matplotlib.pyplot as pyplot
import numpy as numpy
import pylab

# DataSet #1
data = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample1_train.txt")
testdata = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample1_test.txt")

# DataSet #2
# data = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample2_train.txt")
# testdata = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample2_test.txt")

X = [a[0] for a in data]
Y = [a[1] for a in data]
basisVector = [[1, a] for a in X]

w, residuals, _, _ = numpy.linalg.lstsq(basisVector, Y)

# printing output
# ===============
print "Weight Vector"
print w
print ("MSE for training data: %f" %residuals[0])


# Test Data
# =========
testX = [a[0] for a in testdata]
expectedY = [a[1] for a in testdata]
predictedY = [w[0]+w[1]*x for x in testX]

testMSE = 0.0
for i in range(0, len(testX)-1):
    testMSE += pow(predictedY[i] - expectedY[i], 2)
print ("MSE for test data: %f" %testMSE)


# Plotting training data AND testing data
# ==============================
range = numpy.arange(-5,5,0.1)

pyplot.subplot(1, 2, 1)
pyplot.plot(X, Y, "ro")
pyplot.plot(range, w[0]+w[1]*range)
pyplot.title("Training Data: MSE = %f" % residuals[0])

pyplot.subplot(1, 2, 2)
pyplot.plot(testX, expectedY, "ro")
pyplot.plot(range, w[0]+w[1]*range)
pyplot.title("Testing Data: MSE = %f" % testMSE)

pyplot.show();











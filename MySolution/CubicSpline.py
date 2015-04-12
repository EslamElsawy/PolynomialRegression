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
basisVector = [[1, a, pow(a,2), pow(a,3), pow(max(0,a+1),3), pow(max(0,a),3), pow(max(0,a-2),3)] for a in X]

w, residuals, _, _ = numpy.linalg.lstsq(basisVector, Y)

# printing
print "Weight Vector"
print w
print ("MSE: %f" %residuals[0])


# Test Data
# =========
testX = [a[0] for a in testdata]
expectedY = [a[1] for a in testdata]
predictedY = [w[0]+ w[1]*x + w[2]*pow(x,2) + w[3]*pow(x,3) + w[4]*pow(max(0,x+1),3) + w [5]*pow(max(0,x),3) + w[6]*pow(max(0,x-2),3) for x in testX]

testMSE = 0.0
for i in range(0, len(testX)-1):
    testMSE += pow(predictedY[i] - expectedY[i], 2)
print ("MSE for test data: %f" %testMSE)


# Plotting training data AND testing data
# ==============================
range = numpy.arange(-5,5,0.1)

pyplot.subplot(1, 2, 1)
pyplot.plot(X, Y, "ro")
modeloutput = [w[0]+ w[1]*x + w[2]*pow(x,2) + w[3]*pow(x,3) + w[4]*pow(max(0,x+1),3) + w [5]*pow(max(0,x),3) + w[6]*pow(max(0,x-2),3)  for x in range]
pyplot.plot(range, modeloutput)
pyplot.title("Training Data: MSE = %f" % residuals[0])

pyplot.subplot(1, 2, 2)
pyplot.plot(testX, expectedY, "ro")
pyplot.plot(range, modeloutput)
pyplot.title("Testing Data: MSE = %f" % testMSE)

pyplot.show();









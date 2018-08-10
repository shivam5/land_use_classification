"""
    Building model, training and testing over CIFAR data
"""

from __future__ import print_function
import timeit
import gzip
import copy
import numpy
import math
import theano
import theano.tensor as T
from logistic_regression import LogisticRegression
from hidden_layer import HiddenLayer
from loading_data import load_data, load_test_data
from conv_layers import LeNetConvPoolLayer, MyConvPoolLayer, MyConvLayer
import cPickle
import os

actual_labels = load_test_data(1, 1)
n = len(actual_labels)

preds1 = list(numpy.load("../predicted_labels1.npy").reshape(n))
preds2 = list(numpy.load("../predicted_labels2.npy").reshape(n))
preds3 = list(numpy.load("../predicted_labels3.npy").reshape(n))

preds = []

for i in range(len(preds1)):

#    if (preds1[i] == preds2[i]):
#        preds.append(preds1[i])
#    elif (preds1[i] == preds3[i]):
#        preds.append(preds1[i])
#    elif (preds2[i] == preds3[i]):
#        preds.append(preds2[i])
        
    # Appends that model which gets highest accuracy
#    else :
#        preds.append(preds1[i])


#    if (preds2[i] == 3 ):
#        preds.append(preds2[i])

#    elif (preds3[i] == 2 ):
#        preds.append(preds3[i])

#    elif (preds1[i] == 4 ):
#        preds.append(preds1[i])
    
#    elif (preds2[i] == 1 ):
#        preds.append(preds2[i])

#    elif (preds1[i] == 5 ):
#        preds.append(preds1[i])

#    elif (preds2[i] == 0 ):
#        preds.append(preds2[i])

#    else :
#        print("No\n")
#        preds.append(preds1[i])

    if (preds2[i] == 3  ):
        preds.append(preds2[i])
    else :
        preds.append(preds1[i])

    


correct = 0.0
for i in range(n):
    if (preds1[i] == int(actual_labels[i]) ):
        correct += 1.0
        
accuracy = correct/n
print("Number of correctly classified in 1 : ", correct)
print("Test accuracy is in 1", accuracy*100)


correct = 0.0
for i in range(n):
    if (preds2[i] == int(actual_labels[i]) ):
        correct += 1.0
        
accuracy = correct/n
print("Number of correctly classified in 2 : ", correct)
print("Test accuracy is in 2", accuracy*100)


correct = 0.0
for i in range(n):
    if (preds3[i] == int(actual_labels[i]) ):
        correct += 1.0
        
accuracy = correct/n
print("Number of correctly classified in 3 : ", correct)
print("Test accuracy is in 3", accuracy*100)



correct = 0.0
for i in range(n):
    if (preds[i] == int(actual_labels[i]) ):
        correct += 1.0
        
accuracy = correct/n
print("Number of correctly classified : ", correct)
print("Test accuracy is", accuracy*100)

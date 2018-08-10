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

on_unused_input='ignore'

def evaluate_model(learning_rate=0.001, n_epochs=100,
                    nkerns=[16, 40, 50, 60], batch_size=20):
    """ 
    Network for classification of MNIST database

    :type learning_rate: float
    :param learning_rate: this is the initial learning rate used
                            (factor for the stochastic gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer

    :type batch_size: int
    :param batch_size: the batch size for training
    """

    print("Evaluating model")

    rng = numpy.random.RandomState(23455)

    # loading the data1
    datasets = load_test_data(2)

    valid_set_x, valid_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    loaded_params = numpy.load('../saved_models/model2.npy')
    layer4_W, layer4_b, layer3_W, layer3_b, layer2_W, layer2_b, layer1_W, layer1_b, layer0_W, layer0_b = loaded_params

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('Building the model...')

    # Reshape matrix of rasterized images of shape (batch_size, 32 * 32)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (32, 32) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 64, 88))

    # Construct the first convolutional pooling layer:
    # filtering does not reduce the layer size because we use padding
    # maxpooling reduces the size to (32/2, 32/2) = (16, 16)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 16, 16)
    layer0 = MyConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 64, 88),
        p1 = 2, p2 =2,
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2),
        W = layer0_W,
        b = layer0_b
    )

    # Construct the second convolutional pooling layer:
    # filtering does not reduce the layer size because we use padding
    # maxpooling reduces the size to (16/2, 16/2) = (8, 8)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 5, 5)
    layer1 = MyConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 32, 44),
        p1 = 2, p2 =2,
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2),
        W = layer1_W,
        b = layer1_b
    )

    # Construct the third convolutional pooling layer
    # filtering does not reduce the layer size because we use padding
    # maxpooling reduces the size to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[2], 4, 4)
    layer2 = MyConvPoolLayer(
        rng,
        input = layer1.output,
        image_shape = (batch_size, nkerns[1], 16, 22),
        p1 = 2, p2 =2,
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        poolsize=(2, 2),
        W = layer2_W,
        b = layer2_b
    )


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[2] * 4 * 4),
    # or (500, 20 * 4 * 4) = (500, 320) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 8 * 11,
        n_out=800,
        activation=T.tanh,
        W = layer3_W,
        b = layer3_b
    )

    # classify the values of the fully-connected sigmoidal layer
    layer4 = LogisticRegression(input=layer3.output, n_in=800, n_out=6, W = layer4_W, b = layer4_b)

    cost = layer4.negative_log_likelihood(y)

    predicted_output = layer4.y_pred

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    val_model_preds = theano.function(
        [index],
        layer4.prediction(),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        }
    )


    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params



    val_preds = [
        val_model_preds(i)
        for i in range(n_valid_batches)
    ]

    #print(val_preds)
    #preds = numpy(val_preds)
    
    preds = []
    for pred in val_preds:
        for p in pred:
            preds.append(p)
        
    #preds = val_preds.reshape(valid_set_x.get_value(borrow=True).shape[0])
    
    actual_labels = load_test_data(2, 2)
    n = len(actual_labels)
    

    confusion_matrix = numpy.zeros((6,6))
    
    for i in range(n):
        confusion_matrix[int(actual_labels[i])][preds[i]] += 1
    
    print(confusion_matrix)
    
    
    correct = 0.0
    for i in range(n):
        if (preds[i] == int(actual_labels[i]) ):
            correct += 1.0
            
    accuracy = correct/n
    print("Number of correctly classified : ", correct)
    print("Test accuracy is", accuracy*100)    

    # preds = preds.reshape(test_set_x.get_value(borrow=True).shape[0])


#    numpy.save("../val_predicted_labels1", preds)
#    validation_losses = [validate_model(i) for i
#                         in range(n_valid_batches)]

#    test_score = numpy.mean(test_losses)
#    validation_score = numpy.mean(validation_losses)
#    print((' Validation error is %f %%') %
#          (validation_score * 100.))
#    print((' Test error is %f %%') %
#          (test_score * 100.))



if __name__ == "__main__":
    evaluate_model()

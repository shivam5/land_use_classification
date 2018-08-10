"""
    File containing functions for loading datasets 
    and saving them as theano shared variables
"""

import os
from os import listdir
from os.path import isfile, join
import gzip
import six.moves.cPickle as pickle
import numpy
import copy
import random
import theano
import theano.tensor as T
import scipy
from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2


def data_augment_rotate (x):
    '''
        Function for data augmentation,
        it rotates all the images and then returns
        the rasterized vector
    '''

    # Reshaping the input tensor of shape (no_of_examples, avg_height*avg_width*3) 
    # to (no_of_examples, 32, 32, 3) so that it is in proper image format
    shape_x = x.shape
    x = x.reshape(shape_x[0], 3, 32, 32)
    x = numpy.transpose(x, (0,2,3,1))

    # Loads one image, rotates it and saves it
    for i in range(shape_x[0]):

        img = Image.fromarray(x[i,:,:,:], 'RGB')

        # The angle by which the image has to be rotated is chosen 
        # randomly between -30 and 30 degrees
        angle = random.randint(1,60)-30
    
        rot = rotate(img, angle, reshape=False)
        x[i,:,:,:] = rot

    # Reshaping the tesnor of shape (no_of_examples, 32, 32, 3) 
    # to the rasterized format (no_of_examples, avg_height*avg_width*3) 
    x = numpy.transpose(x, (0,3,1,2))
    x = x.reshape(shape_x[0], avg_height*avg_width*3)
    return x


def data_augment_zoom_crop (x):
    '''
        Function for data augmentation,
        it zooms all the images and then crops the 
        image to take the center portion so that the
        size remains the same
    '''

    # Reshaping the input tensor of shape (no_of_examples, avg_height*avg_width*3) 
    # to (no_of_examples, 32, 32, 3) so that it is in proper image format
    shape_x = x.shape
    x = x.reshape(shape_x[0], 3, 32, 32)
    x = numpy.transpose(x, (0,2,3,1))    

    # Loads one image, zooms it, crops the center, and saves it
    for i in range(shape_x[0]):
        img = Image.fromarray(x[i,:,:,:], 'RGB')
        zoom = scipy.ndimage.zoom(img, 1.125, order=1)
        x[i,:,:,:] = zoom[2:-2, 2:-2, :]

    # Reshaping the tesnor of shape (no_of_examples, 32, 32, 3) 
    # to the rasterized format (no_of_examples, avg_height*avg_width*3) 
    x = numpy.transpose(x, (0,3,1,2))
    x = x.reshape(shape_x[0], avg_height*avg_width*3)
    return x


def load_data(number=1):
    ''' 
        Loads the dataset
    '''

    # Join the realtive path to the dataset folder
    dataset_folder = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "Data",
    )

    print('Loading data...')

    # Set the training, test and validation paths
    train_data_images = os.path.join(dataset_folder, "processed_train/images/")
    train_data_label_file = os.path.join(dataset_folder, "processed_train/labels.txt")
    val_data_images = os.path.join(dataset_folder, "processed_val/images/")
    val_data_label_file = os.path.join(dataset_folder, "processed_val/labels.txt")
    test_data_images = os.path.join(dataset_folder, "processed_test/images/")
    test_data_label_file = os.path.join(dataset_folder, "processed_test/labels.txt")


    # print("Calculating average height and width of bounding boxes")

    # heights = []
    # widths = [] 
    # i=0
    # with open(train_data_label_file) as f:
    #     content = f.readlines()
    #     content = [x.strip() for x in content]
    #     for line in content:
    #         if (i%20 == 0):
    #             print (i)
    #         img_name = line.split(":")[0]
    #         img = cv2.imread(os.path.join(train_data_images, img_name))
    #         (height, width, channels) = img.shape
    #         heights.append(height)
    #         widths.append(width)
    #         i += 1
    # avg_height = int(sum(heights)/len(heights))
    # avg_width = int(sum(widths)/len(widths))
    # print("Average height is", avg_height)
    # print("Average width is", avg_width)
    # print("Min height is", min(heights))
    # print("Min width is", min(widths))
    avg_height = 64
    avg_width = 88


    # Loading training data
    if ( os.path.isfile(os.path.join(dataset_folder, "train_data1.npy")) and os.path.isfile(os.path.join(dataset_folder, "train_data2.npy")) and os.path.isfile(os.path.join(dataset_folder, "train_data3.npy")) ) :
        train_data1 = np.load(os.path.join(dataset_folder, "train_data1.npy"));
        train_data2 = np.load(os.path.join(dataset_folder, "train_data2.npy"));
        train_data3 = np.load(os.path.join(dataset_folder, "train_data3.npy"));
    else:
        # Opening the training labels file and reading the training data
        print("Reading training data")
        i=0
        train_data1 = np.zeros((1, avg_height*avg_width+1))
        train_data2 = np.zeros((1, avg_height*avg_width+1))
        train_data3 = np.zeros((1, avg_height*avg_width+1))
        with open(train_data_label_file) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for line in content:
                if (i%20 == 0):
                    print (i)
                img_name = line.split(":")[0]
                label = np.array([int(line.split(":")[1])-1]).reshape((1,1))
                img = cv2.imread(os.path.join(train_data_images, img_name))
                img = cv2.resize(img, (avg_height, avg_width)) 

                img1 = img[:,:,0].reshape(1, avg_height*avg_width)
                img2 = img[:,:,1].reshape(1, avg_height*avg_width)
                img3 = img[:,:,2].reshape(1, avg_height*avg_width)

                instance1 = np.concatenate((img1, label), axis=1)
                instance2 = np.concatenate((img2, label), axis=1)
                instance3 = np.concatenate((img3, label), axis=1)

                train_data1 = np.concatenate((train_data1, instance1), axis=0)
                train_data2 = np.concatenate((train_data2, instance1), axis=0)
                train_data3 = np.concatenate((train_data3, instance1), axis=0)

                i += 1
        train_data1 = train_data1[1:, :]
        train_data2 = train_data2[1:, :]
        train_data3 = train_data3[1:, :]
        np.save( os.path.join(dataset_folder, "train_data1"), train_data1)
        np.save( os.path.join(dataset_folder, "train_data2"), train_data2)
        np.save( os.path.join(dataset_folder, "train_data3"), train_data3)


    # Loading validation data
    if ( os.path.isfile(os.path.join(dataset_folder, "val_data1.npy")) and os.path.isfile(os.path.join(dataset_folder, "val_data2.npy")) and os.path.isfile(os.path.join(dataset_folder, "val_data3.npy")) ) :
        val_data1 = np.load(os.path.join(dataset_folder, "val_data1.npy"));
        val_data2 = np.load(os.path.join(dataset_folder, "val_data2.npy"));
        val_data3 = np.load(os.path.join(dataset_folder, "val_data3.npy"));
    else:
        # Opening the val labels file and reading the val data
        print("Reading validation data")
        i=0
        val_data1 = np.zeros((1, avg_height*avg_width+1))
        val_data2 = np.zeros((1, avg_height*avg_width+1))
        val_data3 = np.zeros((1, avg_height*avg_width+1))
        with open(val_data_label_file) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for line in content:
                if (i%20 == 0):
                    print (i)
                img_name = line.split(":")[0]
                label = np.array([int(line.split(":")[1])-1]).reshape((1,1))
                img = cv2.imread(os.path.join(val_data_images, img_name))
                img = cv2.resize(img, (avg_height, avg_width)) 

                img1 = img[:,:,0].reshape(1, avg_height*avg_width)
                img2 = img[:,:,1].reshape(1, avg_height*avg_width)
                img3 = img[:,:,2].reshape(1, avg_height*avg_width)

                instance1 = np.concatenate((img1, label), axis=1)
                instance2 = np.concatenate((img2, label), axis=1)
                instance3 = np.concatenate((img3, label), axis=1)

                val_data1 = np.concatenate((val_data1, instance1), axis=0)
                val_data2 = np.concatenate((val_data2, instance2), axis=0)
                val_data3 = np.concatenate((val_data3, instance3), axis=0)
                i += 1
        val_data1 = val_data1[1:, :]
        val_data2 = val_data2[1:, :]
        val_data3 = val_data3[1:, :]
        np.save( os.path.join(dataset_folder, "val_data1"), val_data1)
        np.save( os.path.join(dataset_folder, "val_data2"), val_data2)
        np.save( os.path.join(dataset_folder, "val_data3"), val_data3)


    # Loading test data
    if ( os.path.isfile(os.path.join(dataset_folder, "test_data1.npy")) and os.path.isfile(os.path.join(dataset_folder, "test_data1.npy")) and os.path.isfile(os.path.join(dataset_folder, "test_data1.npy")) ) :
        test_data1 = np.load(os.path.join(dataset_folder, "test_data1.npy"));
        test_data2 = np.load(os.path.join(dataset_folder, "test_data2.npy"));
        test_data3 = np.load(os.path.join(dataset_folder, "test_data3.npy"));
    else:
        # Opening the test labels file and reading the test data
        print("Reading test data")
        i=0
        test_data1 = np.zeros((1, avg_height*avg_width+1))
        test_data2 = np.zeros((1, avg_height*avg_width+1))
        test_data3 = np.zeros((1, avg_height*avg_width+1))
        with open(test_data_label_file) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for line in content:
                if (i%20 == 0):
                    print (i)
                img_name = line.split(":")[0]
                label = np.array([int(line.split(":")[1])-1]).reshape((1,1))
                img = cv2.imread(os.path.join(test_data_images, img_name))
                img = cv2.resize(img, (avg_height, avg_width)) 

                img1 = img[:,:,0].reshape(1, avg_height*avg_width)
                img2 = img[:,:,1].reshape(1, avg_height*avg_width)
                img3 = img[:,:,2].reshape(1, avg_height*avg_width)

                instance1 = np.concatenate((img1, label), axis=1)
                instance2 = np.concatenate((img2, label), axis=1)
                instance3 = np.concatenate((img3, label), axis=1)

                test_data1 = np.concatenate((test_data1, instance1), axis=0)
                test_data2 = np.concatenate((test_data2, instance2), axis=0)
                test_data3 = np.concatenate((test_data3, instance3), axis=0)
                i += 1
        test_data1 = test_data1[1:, :]
        test_data2 = test_data2[1:, :]
        test_data3 = test_data3[1:, :]
        np.save( os.path.join(dataset_folder, "test_data1"), test_data1)
        np.save( os.path.join(dataset_folder, "test_data2"), test_data2)
        np.save( os.path.join(dataset_folder, "test_data3"), test_data3)


    # Randomly shuffling the array to ensure no bias in datasets
    numpy.random.shuffle(train_data1)
    numpy.random.shuffle(train_data2)
    numpy.random.shuffle(train_data3)
    
    # Paritioning the image data from labels
    basic_training_data_x1 = train_data1[:, 0:avg_height*avg_width]
    basic_training_data_y1 = train_data1[:, avg_height*avg_width]
    basic_training_data_x2 = train_data2[:, 0:avg_height*avg_width]
    basic_training_data_y2 = train_data2[:, avg_height*avg_width]
    basic_training_data_x3 = train_data3[:, 0:avg_height*avg_width]
    basic_training_data_y3 = train_data3[:, avg_height*avg_width]

    validation_data_x1 = val_data1[:, 0:avg_height*avg_width]
    validation_data_y1 = val_data1[:, avg_height*avg_width]
    validation_data_x2 = val_data2[:, 0:avg_height*avg_width]
    validation_data_y2 = val_data2[:, avg_height*avg_width]
    validation_data_x3 = val_data3[:, 0:avg_height*avg_width]
    validation_data_y3 = val_data3[:, avg_height*avg_width]

    testing_data_x1 = test_data1[:, 0:avg_height*avg_width]
    testing_data_y1 = test_data1[:, avg_height*avg_width]
    testing_data_x2 = test_data2[:, 0:avg_height*avg_width]
    testing_data_y2 = test_data2[:, avg_height*avg_width]
    testing_data_x3 = test_data3[:, 0:avg_height*avg_width]
    testing_data_y3 = test_data3[:, avg_height*avg_width]


    ########################
    ## Standardizing data ##
    ########################

    # Converting to the numpy arrays to type float
    basic_training_data_x1 = numpy.array(basic_training_data_x1, dtype=numpy.float64)
    basic_training_data_x2 = numpy.array(basic_training_data_x2, dtype=numpy.float64)
    basic_training_data_x3 = numpy.array(basic_training_data_x3, dtype=numpy.float64)

    validation_data_x1 = numpy.array(validation_data_x1, dtype=numpy.float64)
    validation_data_x2 = numpy.array(validation_data_x2, dtype=numpy.float64)
    validation_data_x3 = numpy.array(validation_data_x3, dtype=numpy.float64)

    testing_data_x1 = numpy.array(testing_data_x1, dtype=numpy.float64)
    testing_data_x2 = numpy.array(testing_data_x2, dtype=numpy.float64)
    testing_data_x3 = numpy.array(testing_data_x3, dtype=numpy.float64)

    # Standardizing the data (Mean = 0, standard deviation = 0)
    # This is helpful in the convergance of gradient descent
    basic_training_data_x1 -= numpy.mean(basic_training_data_x1 , axis = 0)
    basic_training_data_x1 /= numpy.std(basic_training_data_x1 , axis = 0)
    basic_training_data_x2 -= numpy.mean(basic_training_data_x2 , axis = 0)
    basic_training_data_x2 /= numpy.std(basic_training_data_x2 , axis = 0)
    basic_training_data_x3 -= numpy.mean(basic_training_data_x3 , axis = 0)
    basic_training_data_x3 /= numpy.std(basic_training_data_x3 , axis = 0)

    validation_data_x1 -= numpy.mean(validation_data_x1 , axis = 0)
    validation_data_x1 /= numpy.std(validation_data_x1 , axis = 0)
    validation_data_x2 -= numpy.mean(validation_data_x2 , axis = 0)
    validation_data_x2 /= numpy.std(validation_data_x2 , axis = 0)
    validation_data_x3 -= numpy.mean(validation_data_x3 , axis = 0)
    validation_data_x3 /= numpy.std(validation_data_x3 , axis = 0)

    testing_data_x1 -= numpy.mean(testing_data_x1 , axis = 0)
    testing_data_x1 /= numpy.std(testing_data_x1 , axis = 0)
    testing_data_x2 -= numpy.mean(testing_data_x2 , axis = 0)
    testing_data_x2 /= numpy.std(testing_data_x2 , axis = 0)
    testing_data_x3 -= numpy.mean(testing_data_x3 , axis = 0)
    testing_data_x3 /= numpy.std(testing_data_x3 , axis = 0)

    training_data_x1 = basic_training_data_x1
    training_data_y1 = basic_training_data_y1
    training_data_x2 = basic_training_data_x2
    training_data_y2 = basic_training_data_y2
    training_data_x3 = basic_training_data_x3
    training_data_y3 = basic_training_data_y3

    # # #######################
    # # ## Data augmentation ##
    # # #######################

    # # Creating a copy of the original data
    # aug_data_x = copy.deepcopy(basic_training_data_x)
    # aug_data_y = copy.deepcopy(basic_training_data_y)
    # # Data rotation
    # aug_data_x = data_augment_rotate (aug_data_x)
    # # Stacking the augmented data
    # training_data_x = numpy.vstack([basic_training_data_x, aug_data_x])
    # training_data_y = numpy.concatenate([basic_training_data_y, aug_data_y])
    # training_data = numpy.column_stack( ( training_data_x, training_data_y ) )   
    # # Randomly shuffling data to get a good mix in batches
    # numpy.random.shuffle(training_data)
    # # Separating the image data and labels
    # training_data_x = training_data[:, 0:avg_height*avg_width*3]
    # training_data_y = training_data[:, avg_height*avg_width*3]

    # # Creating a copy of the original data
    # aug_data_x = copy.deepcopy(basic_training_data_x)
    # aug_data_y = copy.deepcopy(basic_training_data_y)
    # # Data zoom crop
    # aug_data_x = data_augment_zoom_crop (aug_data_x)
    # # Stacking the augmented data
    # training_data_x = numpy.vstack([training_data_x, aug_data_x])
    # training_data_y = numpy.concatenate([training_data_y, aug_data_y])
    # training_data = numpy.column_stack( ( training_data_x, training_data_y ) )   
    # # Randomly shuffling data to get a good mix in batches
    # numpy.random.shuffle(training_data)
    # # Separating the image data and labels
    # training_data_x = training_data[:, 0:avg_height*avg_width*3]
    # training_data_y = training_data[:, avg_height*avg_width*3]


    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if number == 1:
        test_set_x, test_set_y = shared_dataset(testing_data_x1, testing_data_y1)
        valid_set_x, valid_set_y = shared_dataset(validation_data_x1, validation_data_y1)
        train_set_x, train_set_y = shared_dataset(training_data_x1, training_data_y1)

    if number == 2:
        test_set_x, test_set_y = shared_dataset(testing_data_x2, testing_data_y2)
        valid_set_x, valid_set_y = shared_dataset(validation_data_x2, validation_data_y2)
        train_set_x, train_set_y = shared_dataset(training_data_x2, training_data_y2)

    if number == 3:
        test_set_x, test_set_y = shared_dataset(testing_data_x3, testing_data_y3)
        valid_set_x, valid_set_y = shared_dataset(validation_data_x3, validation_data_y3)
        train_set_x, train_set_y = shared_dataset(training_data_x3, training_data_y3)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



def load_test_data(number=1, need_preds = 0):
    ''' 
        Loads the dataset for testing and validation
    '''

    # Join the realtive path to the dataset folder
    dataset_folder = os.path.join(
        os.path.split(__file__)[0],
        "..",
        "Data",
    )

    print('Loading data...')

    # Set the test and validation paths
    val_data_images = os.path.join(dataset_folder, "processed_val/images/")
    val_data_label_file = os.path.join(dataset_folder, "processed_val/labels.txt")
    test_data_images = os.path.join(dataset_folder, "processed_test/images/")
    test_data_label_file = os.path.join(dataset_folder, "processed_test/labels.txt")

    avg_height = 64
    avg_width = 88

    if number == 1:
        # Loading validation data
        val_data = np.load(os.path.join(dataset_folder, "val_data1.npy"));
        # Loading test data
        test_data = np.load(os.path.join(dataset_folder, "test_data1.npy"));
    if number == 2:
        # Loading validation data
        val_data = np.load(os.path.join(dataset_folder, "val_data2.npy"));
        # Loading test data
        test_data = np.load(os.path.join(dataset_folder, "test_data2.npy"));
    if number == 3:
        # Loading validation data
        val_data = np.load(os.path.join(dataset_folder, "val_data3.npy"));
        # Loading test data
        test_data = np.load(os.path.join(dataset_folder, "test_data3.npy"));


    # Paritioning the image data from labels
    validation_data_x = val_data[:, 0:avg_height*avg_width]
    validation_data_y = val_data[:, avg_height*avg_width]
    testing_data_x = test_data[:, 0:avg_height*avg_width]
    testing_data_y = test_data[:, avg_height*avg_width]

    if (need_preds == 1):
        #print("The shape of actual preds is", testing_data_y.shape)
        return list(testing_data_y.reshape(testing_data_y.shape[0]))

    if (need_preds == 2):
        #print("The shape of actual preds is", testing_data_y.shape)
        return list(validation_data_y.reshape(validation_data_y.shape[0]))

    ########################
    ## Standardizing data ##
    ########################

    # Converting to the numpy arrays to type float
    validation_data_x = numpy.array(validation_data_x, dtype=numpy.float64)
    testing_data_x = numpy.array(testing_data_x, dtype=numpy.float64)

    # Standardizing the data (Mean = 0, standard deviation = 0)
    # This is helpful in the convergance of gradient descent
    validation_data_x -= numpy.mean(validation_data_x , axis = 0)
    validation_data_x /= numpy.std(validation_data_x , axis = 0)
    testing_data_x -= numpy.mean(testing_data_x , axis = 0)
    testing_data_x /= numpy.std(testing_data_x , axis = 0)

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(testing_data_x, testing_data_y)
    valid_set_x, valid_set_y = shared_dataset(validation_data_x, validation_data_y)

    rval = [(valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval



if __name__ == "__main__":
    load_data()

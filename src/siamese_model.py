# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:17:00 2021

@author: luist
"""

import numpy as np
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Lambda
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K

class Siamese_Model:
    
    """Siamese Neural Network Architecture"""
    
    def __init__(self):
        pass
    
    #//TODO: figure out how to initialize layer weights in keras.
    def weights(self,shape, dtype=None):
    
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    #//TODO: figure out how to initialize layer biases in keras.
    def bias(self,shape, dtype=None):
     
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)
    
    def model(self,dim1,dim2,dim3):
        
        input_shape = (dim1,dim2,dim3)
        
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        
        
        # Siamese Convolutional Neural Networks
        """ -> convolution with 64 filters 10x10;
            -> convolution with ReLU filters (if g(x) > 0 return value else return 0, with g the activation function) - ReLU activation to make all negative value to zero;
            -> convolution with max-pooling layers (escolhe o valor mais elevado em sub-matrizes nxn bem definidas) - Pooling layer is used to reduce the spatial volume of input image after convolution;
            -> convolution with 128 filters 7x7 + ReLU + maxpool;
            -> convolution with 128 filters 2x2 + ReLU + maxpool;
            -> convolution with 256 filters 2x2;
            -> fully-connected layer with 1024 units for classification. 
            
            -> conv 64 filters 10x10, ReLU;
            -> max-pooling layer 
            -> conv 128 filters 7x7, ReLU;
            -> max-pooling layer 
            -> conv 128 filters 2x2, ReLU;
            -> max-pooling layer 
            -> conv 256 filters 2x2, ReLU;
            -> max-pooling layer 
            -> fully connected layer - 1024 units"""
            
        
    
        #build convnet to use in each siamese 'leg'
                                 
        convnet = Sequential()
        convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                           kernel_initializer=self.weights,kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128,(7,7),activation='relu',input_shape=input_shape,
                           kernel_initializer=self.weights,kernel_regularizer=l2(2e-4)))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(128,(2,2),activation='relu',
                           kernel_regularizer=l2(2e-4),kernel_initializer=self.weights,bias_initializer=self.bias))
        convnet.add(MaxPooling2D())
        convnet.add(Conv2D(256,(2,2),activation='relu',kernel_initializer=self.weights,kernel_regularizer=l2(2e-4),bias_initializer=self.bias))
        convnet.add(MaxPooling2D())
        convnet.add(Flatten())
        convnet.add(Dense(1024,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=self.weights,bias_initializer=self.bias))
       
        #encode each of the two inputs into a vector with the convnet
        left_output = convnet(left_input)
        right_output = convnet(right_input)
        
    
        # Layer that computes the absolute difference between the output feature vectors
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1 = L1_layer([left_output, right_output])
    
        # Dense Layer with a sigmoid function that returns the generated similarity score -> output between 0 and 1
        pred = Dense(1,activation='sigmoid',bias_initializer=self.bias)(L1)
    
        # Connect the inputs with the outputs
        net = Model(inputs=[left_input,right_input],outputs=pred)

        return net
    
    
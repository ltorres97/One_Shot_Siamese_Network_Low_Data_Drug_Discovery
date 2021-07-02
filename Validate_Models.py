# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 01:18:37 2021

@author: luist
"""

import numpy.random as rng
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import svm
from keras.layers import Input, Conv1D, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling1D, Dropout, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from sklearn.calibration import CalibratedClassifierCV

class Validate_Models:
    
    "Code to train and test the CNN, SVM, RF and MLP"
    
    def __init__(self,X_train,Xval):
        self.Xval = Xval
        self.X_train= X_train
        self.n_classes,self.n_examples,self.w,self.h = X_train.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape

    def batch_function(self,n,s='train'):
        
        if s == 'train':
            X = self.X_train
        else:
            X = self.Xval
                   
        n_classes, n_examples, w, h = X.shape
        
        """Creates a batch of n pairs, half from the same class and half from different classes to train the network"""
        categories = rng.choice(n_classes,size=(n,),replace=False)
        pairs=[np.zeros((n, w, h,1)) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            index_1 = rng.randint(0,n_examples)
            pairs[0][i,:,:,:] = X[category,index_1].reshape(w,h,1)
            index_2 = rng.randint(0,n_examples)
            
            category_2 = category if i >= n//2 else (category + rng.randint(1,n_classes)) % n_classes
            pairs[1][i,:,:,:] = X[category_2,index_2].reshape(w,h,1)
        
        return pairs, targets
    
    def one_shot_task(self,N,s='val'):
        
        if s == 'train':
            X = self.X_train
        else:
           X = self.Xval
           
        n_classes, n_examples, w, h = X.shape   
        """Creates pairs of a test image and another set to test the network - N-way one-shot learning"""
        categories = rng.choice(n_classes,size=(N,),replace=False)
        indexes = rng.randint(0,n_examples,size=(N,))
        true_category = categories[0]
        example_1, example_2 = rng.choice(n_examples,replace=False,size=(2,))
        test_input = np.asarray([X[true_category,example_1,:,:]]*N).reshape(N,w,h,1)
        support_set = X[categories,indexes,:,:]
        support_set[0,:,:] = X[true_category,example_2]
        support_set = support_set.reshape(N,w,h,1)
        pairs = [test_input,support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        
        return pairs, targets
    

    def train_validate_cnn(self,N, trials, tam, s= 'val'):
        
        """"Training and validation of a single Convolutional Neural Network"""
        
        pairs_train, targets_train = self.batch_function(tam)
        
        list_train = pairs_train[0]
        list_train_2 = pairs_train[1]
    
        pairs_train_list = []
        
        for i in range(len(list_train)):
            sequence_3=[]
            sequence = list_train[i].flatten()
            sequence_2 = list_train_2[i].flatten()
    
            for j in sequence:
                sequence_3.append(j)
            for k in sequence_2:
                sequence_3.append(k)
            
            pairs_train_list.append(np.asarray(sequence_3))
        
        n_corretos = 0
        pairs2train=np.asarray(pairs_train_list).reshape(tam,54*100*2,1)
        targets_train = np.asarray(targets_train).reshape(tam,1)
       
        n_timesteps, n_features, n_outputs = pairs2train.shape[1], pairs2train.shape[2], targets_train.shape[1]
        
        cnn_net = self.cnn_model(pairs2train, targets_train, n_timesteps, n_features, n_outputs)
        
        list_acc=[]
        for n in range(2,N+1):
            for t in range(trials):
            
                pairs_val_2=[]
                pairs_val,targets_val = self.one_shot_task(n,s)
                list_val= pairs_val[0]
                list_val_2 = pairs_val[1]
                
                for i2 in range(len(list_val)):
                
                    sequence_3=[]
                    sequence = list_val[i2].flatten()
                    sequence_2 = list_val_2[i2].flatten()
    
                    for j in sequence:
                        sequence_3.append(j)
                    for k in sequence_2:
                        sequence_3.append(k)
                        
                    pairs_val_2.append(np.asarray(sequence_3))
                                
                pairs2val=np.asarray(pairs_val_2).reshape(n,54*100*2,1)
    
                targets_val = np.asarray(targets_val).reshape(n,1)
    
                prob_vals = cnn_net.predict(pairs2val)
    
                # print("Target:",targets_val)
                # print("Previsão Probabilidade:",prob_vals)
       
                prediction=[]
                for i in prob_vals:
                    prediction.append(i[0])
                
                if np.argmax(prediction) == 0:
                    n_corretos +=1
                    
            percent_correct = (n_corretos / trials)
            
            list_acc.append(percent_correct)
            n_corretos= 0
        
        print("Validation accuracy N = [1, ..., N]::", list_acc)
        return list_acc
      
    
    def cnn_model(self,trainX, trainy,n_timesteps, n_features, n_outputs):
        
        """Architecture of the Convolutional Neural Network"""
        verbose, epochs, batch_size = 1, 10, 50
    
        conv_model = Sequential()
        conv_model.add(Conv1D(64,10,activation='relu', input_shape=(n_timesteps,n_features)))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(128,7,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(128,2,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Conv1D(256,2,activation='relu'))
        conv_model.add(MaxPooling1D())
        conv_model.add(Flatten())
        conv_model.add(Dense(1024,activation="relu"))
        conv_model.add(Dense(n_outputs, activation='sigmoid'))
        
        conv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        conv_model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
        return conv_model
    
    def train_validate_models(self, N, trials, tam, model, batch_size, n_iter, s= 'val'):
        
        """"Processing, Training and Validation of a SVM, Random Forest and a Multi-Layer Perceptron"""
        
        if (model == 'SVM'):
            reg = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2")
            n_iter = 1
            batch_size = tam
        if (model == 'MLP'):
            reg = MLPClassifier(solver='adam', alpha=1e-4,hidden_layer_sizes=(10,5,3), random_state=1)
            n_iter = 1
            batch_size = tam
        if (model == 'RF'):
            reg = RandomForestClassifier(max_depth=None, random_state=0, max_features = 100, n_estimators = 150, warm_start=True)
            
        for i in range(1, n_iter+1):
            
            print("Training iteration:", i)
            
            pairs_train, targets_train = self.batch_function(batch_size)
            
            list_train = pairs_train[0]
            list_test = pairs_train[1]
            
            pairs_train_list = []
            
            for i in range(len(list_train)):
               
                sequence = list_train[i].flatten()
                sequence_2 = list_test[i].flatten()
                sequence_3=[]
                
                for j in sequence:
                    sequence_3.append(j)
                for k in sequence_2:
                    sequence_3.append(k)
                
                pairs_train_list.append(np.asarray(sequence_3))
                
                     
            if (model == 'RF'):
                reg.fit(pairs_train_list, targets_train)
       
        if (model == 'MLP'):
            reg.fit(pairs_train_list, targets_train)              
        
        if (model == 'SVM'):
            reg.fit(pairs_train_list, targets_train)
            calibrator = CalibratedClassifierCV(reg, cv='prefit')
            reg=calibrator.fit(pairs_train_list, targets_train)
            
        n_correct = 0
    
        list_acc=[]
        for n in range(2,N+1):
            for t in range(trials):
            
                pairs_val_list=[]
                pairs_val,targets_val = self.one_shot_task(n,s)
                list_val= pairs_val[0]
                list_val_2 = pairs_val[1]
                
                for i in range(len(list_val)):
                
                    sequence_3=[]
                    sequence = list_val[i].flatten()
                    sequence_2 = list_val_2[i].flatten()
    
                    for j2 in sequence:
                        sequence_3.append(j2)
                    for k2 in sequence_2:
                        sequence_3.append(k2)
                        
                    pairs_val_list.append(np.asarray(sequence_3))
                                                  
                prediction = reg.predict_proba(pairs_val_list)
               
                print("Target:",targets_val)
                print("Previsão Probabilidade:",prediction)
                
                pred_list=[]
                for p in prediction:
                    pred_list.append(p[0])
                
                if np.argmax(pred_list) == 0:
                    n_correct +=1
       
            percent_correct = (n_correct / trials)
            
            list_acc.append(percent_correct)
            n_correct= 0
        
        print("Validation accuracy N = [1, ..., N]:", list_acc)
        
        return list_acc
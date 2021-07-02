# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:45:22 2021

@author: luist
"""

import numpy.random as rng
import numpy as np
import math
class Load_Siamese:
    
    def __init__(self,X_train,Xval):
        self.Xval = Xval
        self.X_train = X_train
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
    
    def one_shot_test(self,model,N,k,s = "val"):
        
        """Evaluates the average accuracy of the network in determining the class of images over a number k of tasks"""
        n_correct = 0
        
        for i in range(k):
            inputs, targets = self.one_shot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct+=1
        percent_correct = (n_correct/ k)
        
        return percent_correct
    
    
    def knn_test(self, N, trials, s='val'):
        
        """Evaluates the average accuracy of the network in determining the class of images over a number k of tasks"""
        n_correct = 0 
        
        for i in range(trials):
            pairs, targets = self.one_shot_task(N,s)
            correct = self.knn_prediction(pairs,targets)
            n_correct += correct
        
        return n_correct/ trials
    
    
    def random_test(self, N, trials, s='val'):
        """Evaluates the average accuracy of the network in determining the class of images over a number k of tasks"""
        n_correct = 0 
        
        for i in range(trials):
            pairs, targets = self.one_shot_task(N,s)
            correct = self.random_prediction(pairs,targets)
            n_correct += correct
        
        return n_correct/ trials
    
        
    def knn_prediction(self,pairs,targets):
        """"Implements the KNN algorithm by considering 1 neighbor as the second element of a pair (k=1)"""
        """returns 1 if nearest neighbour gets the correct answer for a one-shot task given (pairs, targets)"""
        L2_distances = np.zeros_like(targets)
        # print(targets)
        for i in range(len(targets)):
            L2_distances[i] = np.sqrt(np.sum((pairs[0][i].flatten() - pairs[1][i].flatten())**2))
        # print(L2_distances)
        if np.argmin(L2_distances) == 0 and L2_distances[0] != 0:
            return 1
        return 0
    
    
    def random_prediction(self,pairs,targets):
        """"Implements an algorithm to return a random score for each pair"""
        random_predictions = np.zeros_like(targets)
        
        for i in range(len(targets)):
            random_predictions[i] = np.random.uniform(0,1)
        if np.argmax(random_predictions) == np.argmax(targets):
            return 1
        return 0
      
        
      
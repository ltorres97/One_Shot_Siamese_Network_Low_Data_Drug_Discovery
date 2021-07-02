# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 00:44:01 2021

@author: luist
"""

class Validate_one_shot:
    
    def __init__(self):
        pass
    
    "Code to train and test the one-shot Siamese network, KNN and naive models"
    
    def train_validate(self,model, n_iter,loader, time, siamese_net,batch_size,t_start,n_val, evaluate_every, loss_every, n):
        
        
        val_accs , train_accs =  [], []
        best_train = -1        
        best_val = -1 
     
        for i in range(1, n_iter+1):
            
            (inputs,targets) = loader.batch_function(batch_size)
            loss = siamese_net.train_on_batch(inputs, targets)
            
            if i % evaluate_every == 0:
                print("\n ------------- \n")
                print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
                print("Train Loss: {0}".format(loss)) 
                
                
                if model == 'one_shot_siamese':
                    val_acc = loader.one_shot_test(siamese_net, n, n_val)
  
                elif model == 'knn':
                    val_acc = loader.knn_test(n, n_val)
                
                elif model == 'naive':
                    val_acc = loader.random_test(n, n_val)
                    
                train_acc = loader.one_shot_test(siamese_net, n, n_val, s = 'train')

                
                if val_acc >= best_val:
                    print("Current best val: {0}, previous best: {1}".format(val_acc, best_val))
                    best_val = val_acc
                    
                if train_acc >= best_train:
                    print("Current best train: {0}, previous best: {1}".format(train_acc, best_train))
                    best_train = train_acc
    
                    
            if i % loss_every == 0:
                print("iteration {}, training loss: {:.2f},".format(i,loss))
                print("Current best val: {0}".format(best_val)," - N:", n)
                print("Current best train: {0}".format(best_train)," - N:", n)

                print("Tempo decorrido:", (time.time()-t_start)/60.0)
                
                
        print("The final best accuracy value (validation): {0}".format(best_val)," - N:", n)    
        print("The final best accuracy value (training): {0}".format(best_train)," - N:", n) 
   
        val_accs.append(best_val)
        train_accs.append(best_train)
        
        return val_accs, train_accs
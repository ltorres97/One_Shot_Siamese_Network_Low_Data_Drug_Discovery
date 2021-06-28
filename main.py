# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:11:15 2021

@author: luist
"""

from Process_Data import *
import numpy.random as rng
import numpy as np
from Load_Siamese import *
from Siamese_Model import *
import time
from Validate_one_shot import *
from Validate_Models import *

if __name__=='__main__':
    
    process = Process_Data()
    Data_Final = process.process_data('tox21.csv', 'compounds.txt', 8000, 100, 5)
    
    data_train, data_test = train_test_split([i for i in Data_Final], test_size=0.25, random_state=1)
    
    data_train = np.asarray(data_train)
    data_test = np.asarray(data_test)
    
    siamese = Siamese_Model()
    siamese_net = siamese.model(100,54,1)
    
    # Hyper parameters 
    evaluate_every = 100
    loss_every = 1000
    batch_size = 50
    n_iter = 10000
    n_val = 500  #100
    best_val = -1
    best_train = -1
    best_val_knn = -1
    best_val_random = -1
    N = 3 #2,3,4,5,7,10
    
#    model_path = './weights/'
    
    loader = Load_Siamese(data_train,data_test)
    print("Starting training and validation process!")
    print("-------------------------------------")
    t_start = time.time()
    opt = Adam(lr = 0.0001)
    siamese_net.compile(loss="binary_crossentropy",optimizer=opt)
    
    # Uncomment the following commands to run the models
    one_shot_model = Validate_one_shot()
    
    # 1) One-Shot Siamese Model
    # one_shot_model.train_validate('one_shot_siamese',n_iter,loader, time, siamese_net,batch_size,t_start,n_val, evaluate_every, loss_every, N);
    
    # 2) K-Nearest Neighbour
    # one_shot_model.train_validate('knn',n_iter,loader, time, siamese_net,batch_size,t_start,n_val, evaluate_every, loss_every, N);
    
    # 3) Random Model
    one_shot_model.train_validate('naive',n_iter,loader, time, siamese_net,batch_size,t_start,n_val, evaluate_every, loss_every, N);
    
    # 4) CNN, SVM, RF, MLP
    # loader = Validate_Models(data_treino,data_teste)
    
    # 4.1) CNN 
    # loader.train_validate_cnn(N,n_val,len(data_treino))

    # 4.2) Random Forest
    # loader.train_validate_models(N, n_val,len(data_treino),'RF')
    
    # 4.3) SVM
    # loader.train_validate_models(N, n_val,len(data_treino),'SVM')
    
    # 4.4) MLP
    # loader.train_validate_models(N, n_val,len(data_treino),'MLP')
    


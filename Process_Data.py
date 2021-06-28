# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:28:05 2020

@author: luist
"""

from __future__ import print_function
import csv

from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from molvs import validate_smiles
from sklearn.model_selection import train_test_split
import numpy as np
from rdkit.Chem import AllChem
import random
import matplotlib.pyplot as plt

class Process_Data:
    
    def __init__(self):
         pass
    """Butina Clustering algorithm and the Tanimoto distance to perform the clustering"""
    #Define clustering setup
    def ClusterFps(self,fps,cutoff=0.2):
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina
    
        # first generate the distance matrix:
        distances = []
        lenfps = len(fps)
        for i in range(1,lenfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            distances.extend([1-x for x in sims])
    
        # now cluster the data:
        cd = Butina.ClusterData(distances,lenfps,cutoff,isDistData=True)
        return cd
    
    def remove_duplicates(self, x):
        return list(dict.fromkeys(x))
    
    def smiles_encoder(self, smiles):
        """SMILES are encoded into binary matrices based on a dictionary of characters"""
       
        d = {'I': 1, '7': 2, '.': 3, 'l': 4, '8': 5, '(': 6, '[': 7, '2': 8, 'C': 9, 'O': 10, '+': 11, 'F': 12, '9': 13, 'S': 14, ')': 15, 'M': 16, '4': 17, '-': 18, 'N': 19, '1': 20, '3': 21, ']': 22, 'B': 23, 'r': 24, '#': 25, 'P': 26, '=': 27, 'H': 28, 'a': 29, '5': 30, '6': 31, 'g': 32,'c':33, 'n':34, 's': 35, '@':36, '/':37, '\\':38, 'A':39,'i':40,'u':41,'d':42,'o':43,'e':44,'Z':45,'K':46,'V':47,'Y':48,'b':49,'T':50,'G':51,'D':52,'y':53,'t':54}
        #,'@': 33,'o':34,'\\':35, '/':36, 'e':37,'A':38,'Z':39,'K':40, '%':41,'0':42,'i':43,'T':44,'c':45,'s':46,'G':47,'d':48,'n':49,'u':50,'V':51,'R':52,'b':53,'L':54
        X = np.zeros((100,54))
        for i, valor in enumerate(smiles):
            if(d.get(valor) == None):
                print(valor)
            X[i-1, int(d.get(valor))-1] = 1
      
        return X
    
    def process_data(self, csv_data, txt_file, data_size, smile_size, group_size):
        
        """Data is organized and pre-processed to serve as an input to the Siamese model"""
        
        with open(txt_file, "w") as my_output_file:
            with open(csv_data, "r") as my_input_file:
                [my_output_file.write("$$$".join(row)+'\n') for row in csv.reader(my_input_file)]
            my_output_file.close()
        
        with open(txt_file, 'r') as myfile:
          data = myfile.read()
        
        data = data.split("\n")
        
        drug_list=[]
        drug_names = []
        drug_smiles = []
        drug_group = []
        
        count = 0
        for i in data:
            data_splitted = i.split("$$$")
            if len(data_splitted) > 5 and count>0 and count<=data_size:
                drug_names.append(data_splitted[12])
                drug_group.append(data_splitted[2])
                drug_smiles.append(data_splitted[13])
            count+=1  
            
        drug_mol=[]

        for i in drug_smiles:
            if validate_smiles(i) != []:
                drug_smiles.pop(drug_smiles.index(i))
                
        drug_smiles = list(filter(lambda a: len(a) <= smile_size and len(a) > 0, drug_smiles))        
        
        drug_list=[]
        for j in drug_smiles:
            drug_mol.append(Chem.MolFromSmiles(j))
            drug_list.append(len(j))
            
        print(len(drug_mol))
        drug_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in drug_mol]
        
        clusters=self.ClusterFps(drug_fps,cutoff=0.7) #consider a cutoff value of 0.7 for Tanimoto similarity
        
        #Plot the clusters versus the number of elements in each cluster
        index_cluster = [clusters.index(x) for x in clusters]
        count_cluster = [len(x) for x in clusters]
        
        # plt.plot(index_cluster, count_cluster)
        plt.plot(index_cluster, count_cluster,'.g')
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.title("Clustering Tox21 dataset using the Tanimoto Coeficcient")
        plt.show()
        
        Data_Final=[]
        
        i = 0
        """Process the data to form groups with 'group_size' elements each according to the clusters obtained"""
        while i < len(drug_smiles):
            Data = []
            for j in clusters:
                if i in j and drug_smiles[i] != "-":
                    cluster = j
                    
                    while (len(cluster) > group_size):
                        
                        cluster_new = cluster[:group_size]
                                        
                        Data = [self.smiles_encoder(drug_smiles[k]) for k in cluster_new]
                        Data_Final.append(Data)
                        
                        for index in cluster_new:
                            drug_smiles[index] = "-"
                        
                        cluster = cluster[group_size:]
                               
                    while (len(cluster) < group_size):
                        cluster += (random.choice(cluster),)
                        
                    Data = [self.smiles_encoder(drug_smiles[k]) for k in cluster]
                                         
                    Data_Final.append(Data)
                    
                    for index in cluster:
                        drug_smiles[index] = "-"
            
            i+=1 
            
        return Data_Final
    
    def verify_smiles(self,data):
        
         D = []
         for i in data:
           for j in i:
               D.append(j)
    
         return (len(D) != len(set(D)))
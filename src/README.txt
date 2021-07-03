### FILE DESCRIPTION ###

python scripts:
  - load_siamese.py 
        -> load the Siamese Model and organize training and validation batches;
        -> organizes N-way one-shot learning tasks;
        -> evaluates the average accuracy of the network over a number of N tasks across k trials;
        -> implements KNN and naive algortihms.
        
  - process_data.py
        -> Butina Clustering algorithm and the Tanimoto distance to cluster data;
        -> Process the data to form groups with 'group_size' elements each according to the clusters obtained;
        -> SMILES are encoded into binary matrices based on a dictionary of characters;
        -> Data is organized and pre-processed to serve as an input to the Siamese model.
        
  - siamese_model.py
        -> implements the Siamese Neural Network Architecture.
        
  - validate_one_shot.py
        -> code to train and test the one-shot Siamese network, KNN and naive models.
        
  - validate_models.py
        -> code to train and test the CNN, SVM, RF and MLP.
        
  - main.py
        -> main script to run the models and get the results.
data: 
  - tox21.csv
  	-> Tox21 dataset
  - compounds.txt
  	-> .txt file for SMILES processing
	
instructions:
	first, download the file "tox21.csv" in the same directory. 
	second, uncomment the appropriate code lines in "main.py" to run a specific model.
	third, run the file "main.py" to load and prepare the data and train/test the model. 


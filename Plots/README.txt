#plot figure

tanimoto_clustering.png

description: 
  The proposed model aims to predict novel compounds according to the structural similarities observed between molecules.
  The Tanimoto coefficient is used to establish clusters of predictable compounds according to the structural differences observed between 1D sequential structured data. 
 
  - Butina Clustering algorithm and the Tanimoto distance were used to perform the clustering and organize the data into several groups of predictable compounds;
  - A cut-off value of 0.7 for Tanimoto similarity was considered to return the best cluster structure and performance;
  - Tox21 data is processed to form groups with 'group_size' elements each according to the clusters obtained.
  
 Note: The implementation is available in "Process_Data.py" python class.

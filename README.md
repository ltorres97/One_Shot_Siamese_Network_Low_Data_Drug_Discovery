# One-shot Siamese Neural Network Architecture for Low Data Drug Discovery

The application of deep neural networks is an important asset to significantly increase the predictive power when inferring the properties and activities of small-molecules and those of their pharmacological analogues. However, in the traditional drug discovery process where supervised high-quality data is scarce, the lead-optimization step is, inherently, a low-data problem.
\
\
The main objective of this paper is to optimize the discovery of novel compounds based on a reduced set of candidate drugs. We propose the use of a Siamese neural network architecture for one-shot classification, based on Convolutional Neural Networks (CNNs), that learns from a similarity score between two input molecules according to a given similarity function.
\
\
Using a one-shot learning strategy, we only need a few instances per class for the network's training and a small amount of data and computational resources to build an accurate model. The results of this study show that a one-shot-based classification using a Siamese neural network allowed us to outperform the graphical convolution and the state-of-the-art models in the accurate and reliable prediction of novel compounds given the lack of biological data available for lead discovery.

# Requirements

We used Python 3.7.3 and Keras with Tensorflow back-end to develop the main Siamese Model and the standart convolutional model (CNN). Sckit-learn was used to implement some of the other models for comparison purposes as Random Forest, SVM and Multi-Layer Perceptron.
RDKit was used to pre-process the SMILEs Strings and access SMILEs fingerprints.

In order to get started you will need:


- Python 3.7.3
- Keras
- Tensorflow 1.14
- RDKit
- Scikit-learn
- Numpy
- Pandas
- molvs


# Our Approach
In this study, starting from a reduced set of candidate molecules, we adapt the convolutional neural network (CNN) to predict novel compounds according to the structural dissimilarities observed between molecules. The proposed model accepts different pairs of molecules and learns a similarity function which returns a similarity score between two input molecules. Thus, according to the learned similarity rule the network predicts the similarity score in one-shot.
\
\
We introduce an approach compatible with the set of pairs of compounds provided, a Siamese neural network built upon two parallel and identical convolutional neural networks. The model learns a similarity function and returns a distance metric applied to the output feature vectors from both Siamese twins given a pair of input molecules. 

[![](https://github.com/ltorres97/One_Shot_Siamese_Network_Low_Data_Drug_Discovery/blob/master/siamese_net.png?raw=true)](https://github.com/ltorres97/One_Shot_Siamese_Network_Low_Data_Drug_Discovery/blob/master/siamese_net.png?raw=true)
# Methodology
## Model Overview

We introduce a model that accepts data organized in pairs, a Siamese neural network - two parallel and identical convolutional neural networks (CNNs). Both Siamese twins are indistinguishable, being two copies of the same network, sharing the same set of parameters. 
\
These parallel networks reduce their respective inputs to increasingly smaller tensors as we progress to high-level layers. The difference between the output feature vectors is used as an input to a linear classifier, a similarity function. 

## Siamese Neural Network Architecture

The main model consists of a Siamese Neural Network based on convolutional neural networks (CNNs). The network's input consists of two twin input vectors with a corresponding hidden vector in each convolutional layer. The model architecture that maximizes performance is the one whose number of convolutional layers is 4, whose number of filters in each layer is a multiple of 16 and in which the corresponding output features maps are applied to a ReLu activation function and to a maxpooling layer.
\
\
Four convolutional layers are interspersed with 3 max-pooling layers of stride 2. A ReLU activation is applied to the first layers to accelerate the process of convergence and a sigmoid function in the last layer condenses the final score in a value between 0 and 1,
\
\
The output feature maps of the last convolutional layer serve as an input to a fully connected (FC) layer with 1024 units. The FC layer learns a similarity function and computes the distance between both embedded outputs of the Siamese Twins. The absolute difference between both flattened embeddings serve as input to a sigmoid function to return the predicted similarity score.
\
\
A binary cross entropy loss between the predictions and targets is applied.
### Hyperparameter Optimization:

Hyperparameter optimization does not consider adding any additional convolutional layers and involves the fine-tuning of several network parameters. Four main network parameters were selected for optimization: the number of convolutional filters, the size of the convolutional filters, the number of units and the learning rate. 
\
\
All neural network weights and biases are initialized from a normal distribution with mean 0 and 0.5, respectively. Additionally, we considered a standard deviation for weights and biases of 0.01. 
\
\
K-fold cross-validation is replaced by early stopping and model checkpoint to save the model parameters which return the best accuracy results on the validation set. Hence, the performance is evaluated over a certain number of epochs within a tolerance of 0.0001.
\
\
The size of the convolutional filters varies from 2x2 to 20x20, while the number of convolutional filters varies from 16 to 256 in multiple of 16 and from 256 and 1024 for the FC layer. The layerwise learning rate parameter was held in an interval between 0.000001 and 0.1 considering steps of 0.1.


## Pairwise Training

Compounds are organized in pairs, considering a task-specific support set and a disjoint query set. For each task, we consider a training set with half of the pairs of the same class and half of the pairs of different classes. Consequently, the data set size increases with the total number of possible combinations for the pairs of compounds available for training. 
\
\
Consequently, the data set size increases with the number of classes of a square factor and with the number of examples per class of a linear factor to increase the predictive power and prevent overfitting.
\
\
The maximum number of combinations for the pairs of molecules is calculated to obtain a ratio of 1:1 for both pair configurations.

## One-Shot Learning Approach: Training and Testing

The reduced amount of biological data for training led us to adopt a new strategy to predict novel compounds using the proposed model. We consider a one-shot classification strategy to demonstrate the discriminative power of the learned features.
\
Note that for every pair of inputs, our model generates a similarity score between 0 and 1 in one shot. Therefore, to evaluate whether the model is really able to recognize similar molecules and distinguish dissimilar ones, we use an N-way one shot learning strategy. The same molecule is compared to N different ones and only one of those matches the original input. Thus, we get N different similarity scores {p_1, ..., p_N} denoting the similarity between the test molecules and those on the support set. We expect that the pair of compounds with the same class gets the maximum score and we treat this as a correct prediction. If we repeat this process across multiple trials, we are able to calculate the network accuracy as the percentage of correct predictions.
\
\
In practice, for model validation we select a test instance that is compared to each one of the molecules in the support set. The support set consists of set of molecules representing each class selected at random whenever a one-shot learning task is performed.
\
Subsequently, we pair the test instance with each compound in the support set and check which one gets the highest similarity score. We conclude that the prediction is correct if the maximum score corresponds to the pair of molecules of the same class. Thus, in each trial, we organize the pairs for validation so that the first pair is a pair of instances of the same class, with the remaining pairs formed by compounds of different classes.
\
\
The final prediction corresponds to the pair of compounds which returns the highest similarity score in a one-shot trial. The value N refers to the number of pairs of molecules in comparison at each trial in a N-way one-shot task. Note that, increasing the number of comparisons N, the more challenging it becomes to make a correct prediction and lower is the final accuracy of the model. Consequently, it is more difficult to return the maximum similarity score for the first pair due to the presence of a greater number of pairs in comparison at each trial. This behavior increases the chances of a failed prediction as we progress to higher values of N, which leads to a significant drop in the final accuracy values.


# Model Comparison and Results

The comparison of a given complex model with a set of simpler base models is a common strategy when assessing performance. In this case, compounds are represented by binary matrices so it was necessary to reduce their dimension by converting them to pairs of flattened vectors. This representation led to a consistent model evaluation and to a meaningful performance comparison between different models.
\
\
In the implementation of those models, we provide a  set  of  drug  pairs and we merge and represent them as concatenated flattened vectors. We consider a training set in which  half  are  pairs  of  the  same  class  and  another  half  of different classes. To ensure consistency in the performance evaluation,  the  accuracy  of  the  model  was  determined using  the  same  strategy mentioned previously.  We  define  a  set  of  N-way  one-shot  tasks  that  allows  the comparison of a set of  N concatenated pairs across 500 trials. 

## Accuracy Results


| Model | #2-way  | #3-way  | #4-way | #5-way  | #7-way  | #10-way |
| ------- | --- | --- | --- | --- | --- | --- |
| Siamese val | 94% | 90% | 84% | 78% | 70% | 65% |
| Siamese train | 95% | 92% | 86% | 84% | 72% | 70% |
| KNN | 70% | 55% | 49% |  43% | 36% | 30% |
| Random | 61% | 43% | 34% | 31% | 22% | 19% |
| SVM | 56% | 42% | 30% | 24% | 16% | 12% |
| Random Forest | 71% | 58% | 60% | 44% | 34% | 20% |
| CNN |  81% | 70% | 58% | 46% | 41% | 39% |
| MLP | 78% | 60% | 36% | 34% | 22% | 13% |

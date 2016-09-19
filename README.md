This project contains files to extract training data and test data from MNIST
data set. The data has been trained with 4-layer neural network. The 2 hidden
layers contain 300 units in each layer. I have used scipy.optimize.fmin_cg 
function to minimize the cost function of algorithm. Back propagation algorithm
is used for calculating gradients. I have run the algorithm for 250 iteratiions 
as of now and I have achieved 93.7 accuracy%. The weights with which this accuracy 
is achieved is saved in a "weights.npy" file.

Please ignore the files with '.pyc' extensions 

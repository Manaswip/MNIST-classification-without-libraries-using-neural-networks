import numpy as np
import CostFunction as n
import RandomInitializeWeights as p
import scipy.io as sio
from scipy import optimize as o
import ExtractingData as e
import ExtractingTestData as t
import WeightsGradient as g
import sigmoid as s

#Loading training data
X,y,rows,cols = e.ExtractingData()
#Loading test data
X_test,y_test,rows_test,cols_test =  t.ExtractingData()
#number of training samples
m = X.shape[0]
#number of testing samples
m_test = X_test.shape[0]
# Initializing variables
input_layer_size  = rows*cols;  # 28x28 Input Images of Digits
hidden_layer_size = 300;   # 300 hidden units in 2 layers
num_labels = 10;          # 10 labels, from 0 to 9   

#Initializing weights to random values
#Theta1 = p.randInitializeWeights(input_layer_size,hidden_layer_size)
#Theta2 = p.randInitializeWeights(hidden_layer_size,hidden_layer_size)
#Theta3 = p.randInitializeWeights(hidden_layer_size,num_labels)
#Converting random values of all layers to a single column matrix to use scipy optimize functions
#nn_params = np.concatenate([(np.transpose(Theta1)).ravel(),(np.transpose(Theta2)).ravel(),(np.transpose(Theta3)).ravel()])




#Regularization parameter
lambda1 = 1;

#J = n.nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1);

# Short hand notation for the cost function to be minimized, returns the cost
costFunct = lambda p:n.nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1);
# Short hand notation for the cost function to be minimized, returns the gradient
gradFunct = lambda p:g.gradientCal(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1);

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
for j in range(0,5):
#Initializing weights to parameters learnt after running the algorithm for few iterations
	nn_params = np.load('weights.npy')
	params = o.fmin_cg(costFunct,nn_params,fprime=gradFunct,maxiter=50)
#save the parameters learnt after running the code for few iterations
	np.save('weights',params)


#Obtain weights back from params
T1 = np.reshape(params[0:hidden_layer_size * (input_layer_size + 1)],
    	((input_layer_size + 1),hidden_layer_size));

T2 = np.reshape(params[(hidden_layer_size * (input_layer_size + 1)):(hidden_layer_size * (input_layer_size + 1))+((hidden_layer_size+1)*hidden_layer_size)],
    ((hidden_layer_size + 1),hidden_layer_size));

T3 = np.reshape(params[(hidden_layer_size * (input_layer_size + 1))+((hidden_layer_size+1)*hidden_layer_size):],
    ((hidden_layer_size + 1),num_labels));

T1 = np.transpose(T1)
T2 = np.transpose(T2)
T3 = np.transpose(T3)

#Adding bias unit to the input 
X_test = np.concatenate((np.ones((m_test,1)),X_test),axis=1)

first_layer_activation = X_test;

#Second layer activation function
second_layer_activation = s.sigmoid(np.dot(T1,np.transpose(X_test)));
# appending bias unit
second_layer_activation = np.concatenate((np.ones((1,m_test)),second_layer_activation));
#Activation in third layer
third_layer_activation =  s.sigmoid(np.dot(T2,second_layer_activation));
#appending bias unit
third_layer_activation = np.concatenate((np.ones((1,m_test)),third_layer_activation));
#Hypothesis function
hypFunction = s.sigmoid(np.dot(T3,third_layer_activation));

predict = np.argmax(hypFunction,0)

predict = predict[np.newaxis]
predict = np.transpose(predict)

#How many were predicted correctly
A = (predict == y_test)
np.uint8(A)

#94.48 accuracy 450 iterations
print "Training Accuracy:", np.mean(A)*100
from __future__ import division
import numpy as np
import sigmoid as s

""" Calculating cost function similar to logistic regression
 with out any special libraries 
"""

def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1):

# Getting back the weights
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
    	((input_layer_size + 1),hidden_layer_size));

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):(hidden_layer_size * (input_layer_size + 1))+((hidden_layer_size+1)*hidden_layer_size)],
    ((hidden_layer_size + 1),hidden_layer_size));

    Theta3 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1))+((hidden_layer_size+1)*hidden_layer_size):],
    ((hidden_layer_size + 1),num_labels));

    Theta1 = np.transpose(Theta1);
    Theta2 = np.transpose(Theta2);
    Theta3 = np.transpose(Theta3);
    new_unrolled_Theta = np.concatenate([(np.transpose(Theta1[:,1:])).ravel(),(np.transpose(Theta2[:,1:])).ravel(),(np.transpose(Theta3[:,1:])).ravel()])
# Number of trainig data sets
    m = X.shape[0];
# Concatenating bias unit
    X = np.concatenate((np.ones((m,1)),X),axis=1)

# Number of features
    n = X.shape[1];
# transforming y to new matrix with 10xm size
    new_y = np.zeros((num_labels,m))

    for i in range(0,m):
        new_y[y.item(i),i] = 1;

    first_layer_activation = X;
#Second layer activation function
    second_layer_activation = s.sigmoid(np.dot(Theta1,np.transpose(X)));
# appending bias unit
    second_layer_activation = np.concatenate((np.ones((1,m)),second_layer_activation));
#Activation in third layer
    third_layer_activation =  s.sigmoid(np.dot(Theta2,second_layer_activation));
#appending bias unit
    third_layer_activation = np.concatenate((np.ones((1,m)),third_layer_activation));
#Hypothesis function
    hypFunction = s.sigmoid(np.dot(Theta3,third_layer_activation));

    first_half = np.sum(np.multiply(new_y,np.log(hypFunction)));
    second_half = np.sum(np.multiply((1-new_y),np.log(1-hypFunction)));
#Calculating cost
    J = ((-1.0/m)*(first_half+second_half)) + (lambda1/(2*m) *(np.sum(np.multiply(new_unrolled_Theta,new_unrolled_Theta))));

    return J















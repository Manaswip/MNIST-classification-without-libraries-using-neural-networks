from __future__ import division
import numpy as np
import sigmoid as s
import scipy.io as sio
import sigmoidGradient as si

""" This program returns gradients of weights
"""

def gradientCal(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1):
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
    Theta1_grad = np.zeros(Theta1.shape);
    Theta2_grad = np.zeros(Theta2.shape);
    Theta3_grad = np.zeros(Theta3.shape);
# Concatenating bias unit
    X = np.concatenate((np.ones((m,1)),X),axis=1)

# Number of features
    n = X.shape[1];

    new_y = np.zeros((num_labels,m))

    for i in range(0,m):
        new_y[y.item(i),i] = 1;
  #  	new_y[y.item(i),i] = 1;

#Calculating delta terms using back propogation
    delta1_grad = np.zeros(Theta1.shape);
    delta2_grad = np.zeros(Theta2.shape);
    delta3_grad = np.zeros(Theta3.shape);

#back propogation algorithm
    for i in range(0,m):
    	a1 = X[i,:][np.newaxis]
    	a2 = s.sigmoid(np.dot(Theta1,np.transpose(a1)));
    	a2 = np.concatenate((np.ones((1,1)),a2));
    	a3 = s.sigmoid(np.dot(Theta2,a2));
        a3 = np.concatenate((np.ones((1,1)),a3));
        a4 = s.sigmoid(np.dot(Theta3,a3));

    	delta4 = a4 - np.transpose(new_y[:,i][np.newaxis]);
    	delta = np.dot((np.transpose(Theta3)),delta4);
    	delta3 = np.multiply(delta[1:],si.sigmoidGradient(np.dot(Theta2,a2)));
        delta = np.dot((np.transpose(Theta2)),delta3);
        delta2 = np.multiply(delta[1:],si.sigmoidGradient(np.dot(Theta1,np.transpose(a1))));

    	delta1_grad = delta1_grad + np.dot(delta2,a1);
        delta2_grad = delta2_grad + np.dot(delta3,np.transpose(a2));
    	delta3_grad = delta3_grad + np.dot(delta4,np.transpose(a3));

    Theta1_grad = (1/m) * delta1_grad;
    Theta2_grad = (1/m) * delta2_grad;
    Theta3_grad = (1/m) * delta3_grad;
#gradient of 3 layers
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (Theta1[:,1:]) * (lambda1/m);
    Theta2_grad[:,1:] = Theta2_grad[:,1:] +  (Theta2[:,1:]) * (lambda1/m);
    Theta3_grad[:,1:] = Theta3_grad[:,1:] +  (Theta3[:,1:]) * (lambda1/m);
#concatenate 3 layers gradient to one vector
    grad = np.concatenate([(np.transpose(Theta1_grad)).ravel(),(np.transpose(Theta2_grad)).ravel(),(np.transpose(Theta3_grad)).ravel()]);

    return grad















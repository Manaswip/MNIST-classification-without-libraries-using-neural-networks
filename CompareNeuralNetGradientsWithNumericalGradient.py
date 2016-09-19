from numpy import linalg as LA
import DebugInitialWeights as d
import CostFunction as n
import ComputeNumericalGradient as c
import numpy as np
import WeightsGradient as g

""" In order to check if the code is working as expected for cost function, 
calculating gradients of weights, intialize data which is comparitively
small and find gradients using WeightsGradient and comapre
with numerical gradient of data, if the difference is relatively slow then
code is working as expected
"""

input_layer_size = 3;
hidden_layer_size = 5;
num_labels = 3;
m = 5;
lambda1 = 0;

#We generate some 'random' test data
Theta1 = d.debugInitializeWeights(hidden_layer_size, input_layer_size);
Theta2 = d.debugInitializeWeights(hidden_layer_size,hidden_layer_size);
Theta3 = d.debugInitializeWeights(num_labels, hidden_layer_size);
# Reusing debugInitializeWeights to generate X
X  = d.debugInitializeWeights(m, input_layer_size - 1);
y  = np.mod(np.arange(1,m+1),num_labels) ;


nn_params = np.concatenate([(np.transpose(Theta1)).ravel(),(np.transpose(Theta2)).ravel(),(np.transpose(Theta3)).ravel()])

costFunct = lambda p:n.nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1);
gradFunct = lambda p:g.gradientCal(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda1);
grad = gradFunct(nn_params);

numgrad = c.computeNumericalGradient(costFunct, nn_params);

diff = LA.norm(numgrad-grad)/LA.norm(numgrad+grad);

print diff


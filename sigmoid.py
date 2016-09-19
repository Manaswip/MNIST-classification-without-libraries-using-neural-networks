import numpy as np
def sigmoid(z):
#given a matrix,scalar or vector the sigmoid function of them is returned
	g= np.divide(1,(1+ np.exp(-z)))
	return g;
	
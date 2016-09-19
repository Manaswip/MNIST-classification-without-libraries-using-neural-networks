import numpy as np

def debugInitializeWeights(fan_out, fan_in):

	W = np.zeros((fan_out,fan_in+1))

	Z = np.reshape(np.sin(np.arange(1,(W.size+1))),(fan_in+1,fan_out))/10;

	return np.transpose(Z);
import numpy as np

""" Compute numerical gradient of data which is dy/dx = (y2-y1)/(x2-x1)
"""

def computeNumericalGradient(J,Theta):

	numgrad = np.zeros((Theta.shape))
	per = np.zeros((Theta.shape))

	a = 1e-4;

	for i in range(Theta.size):
		per[i] = a;
		loss1 = J(Theta-per);
		loss2 = J(Theta+per);
		numgrad[i] = (loss2-loss1) / (2*a);
		per[i]=0;

	return numgrad;




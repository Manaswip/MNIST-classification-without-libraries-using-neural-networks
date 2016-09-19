import numpy as np

# Initialize weights randomyl
def randInitializeWeights(L_in,L_out):
	W = np.zeros((L_out,L_in+1))
	Init_epsilon = 0.12;
#initializing weights in the range of (-init_epsilon,init_epsilon)
	W = np.random.rand(L_out,1+L_in)*2*Init_epsilon - Init_epsilon

	return W	
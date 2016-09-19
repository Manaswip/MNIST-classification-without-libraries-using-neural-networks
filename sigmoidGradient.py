import sigmoid as s
import numpy as np

# Derivative of sigmoid ativation function
def sigmoidGradient(z):
	x = s.sigmoid(z);
	g = np.multiply(x,1-x);
	return g;
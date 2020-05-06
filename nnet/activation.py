
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def sigmoid(z):
	"""Calculates sigmoid values for tensors

	"""
	result = 1/(1+torch.exp(-z))
	return result.float()

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
	"""Calculates derivative of sigmoid function

	"""
	xe=sigmoid(z)
	grad_sigmoid = xe.float()*(1-xe).float()
	return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
	"""Calculates stable softmax (minor difference from normal softmax) values for tensors

	"""
	#xc=torch.exp(x-torch.max(x))
	#stable_softmax = xc / torch.sum(xc)

	x=x.exp()
	for i in range(len(x)):
		x[i]=x[i]/torch.sum(x[i])
	return x
	#return stable_softmax.float()

if __name__ == "__main__":
	pass
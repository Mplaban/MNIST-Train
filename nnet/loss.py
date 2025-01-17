
# NOTE: You can only use Tensor API of PyTorch

import torch
import numpy as np
from math import e
from nnet import activation

# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
	"""Calculates cross entropy loss given outputs and actual labels

	"""
	m = labels.shape[0]
	p = activation.softmax(outputs)
	log_likelihood = -torch.log(p[range(m),labels])
	creloss = torch.sum(log_likelihood) / m

	return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
	"""Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
	
	"""

	m = labels.shape[0]
	grad = activation.softmax(outputs)
	grad[range(m),labels] -= 1
	avg_grads = grad/m

	return avg_grads

if __name__ == "__main__":
	pass
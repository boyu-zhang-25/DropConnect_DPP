from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import matplotlib.pyplot as plt
from scipy.special import erf, expit

# from dpp_sample import *
from teacher_dataset import *
from dpp_sample_expected import *

def dgdx_erf(x, sd=None):
	"""
	Parameters:
	-----------
	sd : None or scalar
		if not None, standard deviation of the Gaussian noise that is injected.
	"""
	zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
	return np.exp(-(x/np.sqrt(2) + zeta)**2) * np.sqrt(2 / np.pi)

def g_erf(x, sd=None):
	"""
	Sigmoidal activation function used for both teacher and student.

	We use the error function with this particular scaling because it makes
	analytical calculations more convenient.

	Parameters:
	-----------
	sd : None or scalar
		if not None, standard deviation of the Gaussian noise that is injected.
	"""
	zeta = 0 if sd is None else sd * rnd.randn(*x.shape)
	return erf(x / np.sqrt(2) + zeta)


def gradient(w, xis, ys, g=g_erf, dgdx=dgdx_erf, normalise=False):
	"""
	Returns the gradient value for vanilla online gradient descent.

	Parameters:
	-----------
	w : (K, N)
	xis : (bs, N)
		the inputs used in this step, where bs is the batchsize
	ys :
		the teacher's outputs for the given inputs
	g, dgdx :
		student's activation function and its derivative
	normalise :
		True if the the output of the SCM is normalised by K, otherwise False.
	"""
	bs, N = xis.shape

	error = np.diag(ys - phi(w, xis, g, normalise))

	print('output:', phi(w, xis, g, normalise))
	print('error: ', error)

	if normalise:
		error /= w.shape[0]

	return 1. / bs * dgdx_erf(w @ xis.T / np.sqrt(N)) @ error @ xis

def phi(weights, xis, g=g_erf, normalise=False):
	"""
	Computes the output of a soft committee machine with the given weights.

	Parameters:
	-----------
	w : (r, N)
		weight matrix, where r is the number of hidden units
	xis : (batchsize, N)
		input vectors
	g : activation function
	normalise :
		True if the the output of the SCM is normalised by the number of hidden
		units.
	"""
	K, N = weights.shape
	phi = np.sum(g(weights @ xis.T / np.sqrt(N)), axis=0)
	# phi = np.sum(expit(weights @ xis.T), axis=0)
	return phi / K if normalise else phi

'''
inputs = np.load('inputs.npy')
labels = np.load('labels.npy')

w1 = np.load('student_w1.npy')

for i in range(inputs.shape[1]):

	x = np.expand_dims(inputs[:, i], axis = 0)
	label = labels[i]

	print('x:', x.shape)
	print('label:', label)

	grad = gradient(w1, x, label, g=g_erf, dgdx=dgdx_erf, normalise=False)
	# print('w1 grad:', grad)

	w1_torch = np.load('w1_torch.npy')
	print(grad / w1_torch)
'''


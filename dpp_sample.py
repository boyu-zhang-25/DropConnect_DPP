from dppy.finite_dpps import FiniteDPP
import numpy as np
from sklearn.metrics import pairwise_distances as pd

def sample_dpp(weighted_input,beta=0.3,k=3):

	ker = np.exp(-beta*(pd(weighted_input.T,metric='l2'))**2)
	assert(ker.shape[0]==weighted_input.shape[1])
	DPP = FiniteDPP('likelihood',**{'L':ker})
	DPP.sample_exact_k_dpp(size=k)
	x = list(DPP.list_of_samples)[0]
	assert(len(x)==k)
	return x


def create_weight(input,weight):
	return input*(weight.T)[:,np.newaxis]


#input = array of shape (num_inp * inp_dimension)
#weight = array of shape (inp_dim * hid_dim)
#k = the number of incoming edges to keep for each hidden node

def dpp_sample_edge(input,weight,beta=0.3,k=3):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	weighted_input_mat = create_weight(input,weight)
	samples = []

	for w_inp in weighted_input_mat:

		samples.append(sample_dpp(w_inp,k=k))

	mask = np.zeros((inp_dim,hid_dim))
	for j in range(len(samples)):
		for i in samples[j]:
			mask[i][j] = 1

	return mask


def dpp_sample_node(input,weight,beta=0.3,k=2):

	inp_dim = weight.shape[0]

	weighted_input = np.dot(input,weight)
	sample = sample_dpp(weighted_input,k=k)

	mask = np.zeros((inp_dim,hid_dim))
	for hid_node in sample:
		mask[:,hid_node] = np.ones(inp_dim)
	return mask


'''
N, inp_dim, hid_dim = 10, 5, 3
input = np.random.normal(size=(N,inp_dim))
weight = np.random.normal(size=(inp_dim,hid_dim))
print(input)
print('-----')
print(weight)
print('-----')
x=dpp_sample_edge(input,weight)
print(x.shape)
print(x)
'''

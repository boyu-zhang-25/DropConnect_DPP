import numpy as np
from sklearn.metrics import pairwise_distances as pd
import pickle as pkl
from dppy.finite_dpps import FiniteDPP

def create_kernel(weighted_input,beta):
	return np.exp(-beta*(pd(weighted_input.T,metric='l2'))**2)

def sample_dpp(kernel,k):
	DPP = FiniteDPP('likelihood',**{'L':kernel})
	DPP.sample_exact_k_dpp(size=k)
	x = list(DPP.list_of_samples)[0]
	assert(len(x)==k)
	return x

def create_weight(input,weight):
	print(input.shape, weight.T.shape, (weight.T)[:,np.newaxis].shape)
	return input*(weight.T)[:,np.newaxis]


#input = array of shape (num_inp * inp_dimension)
#weight = array of shape (inp_dim * hid_dim)
#k = the number of incoming edges to keep for each hidden node

def create_edge_kernel(input, weight, beta, dataset):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	weighted_input_mat = create_weight(input,weight)
	print(weighted_input_mat.shape)
	ker_list = []
	for w_inp in  weighted_input_mat:
		ker_list.append(create_kernel(w_inp,beta))
	file_name = dataset + '_ker_list.pkl'
	with open(file_name, 'wb') as f:
		pkl.dump(ker_list,f)
	return ker_list




def dpp_sample_edge(input, weight, beta, k, dataset, load_from_pkl = True):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	if load_from_pkl:
		file_name = '../' + dataset + '_ker_list.pkl'
		ker_list = pkl.load(open(file_name, 'rb'))
		print('loaded kernel', file_name, len(ker_list), ker_list[0].shape)
	else:
		ker_list = create_edge_kernel(input, weight, beta, dataset)
		print('created kernel', str(dataset + '_ker_list.pkl'))
	samples = []
	for iter_num,ker in  enumerate(ker_list):
		# print(iter_num,'sampling from DPP')
		samples.append(sample_dpp(ker,k))

	mask = np.zeros((inp_dim,hid_dim))
	for j in range(len(samples)):
		for i in samples[j]:
			mask[i][j] = 1

	return mask


def dpp_sample_node(input,weight,beta,k):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	weighted_input = np.dot(input,weight)
	ker = create_kernel(weighted_input,beta)
	sample = sample_dpp(ker,k)

	mask = np.zeros((inp_dim,hid_dim))
	for hid_node in sample:
		mask[:,hid_node] = np.ones(inp_dim)
	return mask

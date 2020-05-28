import numpy as np
from sklearn.metrics import pairwise_distances as pd
import pickle as pkl
from dppy.finite_dpps import FiniteDPP
from sklearn.linear_model import LinearRegression



def create_kernel_ts(weighted_input,beta):
	N = weighted_input.shape[0]
	return (1.0/N)*np.dot(weighted_input.T,weighted_input)


def sample_dpp_multiple_ts(kernel, k, num_masks):

	'''
	return a list of length num_masks
	each element is a numpy array of length k as the sampling result
	'''
	DPP = FiniteDPP('likelihood', **{'L' : kernel})
	for _ in range(num_masks):
		DPP.sample_exact_k_dpp(size = k)
	return DPP.list_of_samples

def create_weight(input,weight):
	print('creating weight...')
	print('input.shape: {}, weight.T.shape: {}, (weight.T)[:,np.newaxis].shape: {}'.format(input.shape, weight.T.shape, (weight.T)[:,np.newaxis].shape))
	return input*(weight.T)[:, np.newaxis]


# create kernel based on all edges incoming to one node
def create_edge_kernel_ts(input, weight, beta, dataset):

	'''
	input = array of shape (num_inp * inp_dimension)
	weight = array of shape (inp_dim * hid_dim)
	'''

	weighted_input_mat = create_weight(input, weight)
	print('weighted_input_mat.shape:', weighted_input_mat.shape)

	# one kernel per node (all incoming edges)
	ker_list = []
	for w_inp in  weighted_input_mat:
		ker_list.append(create_kernel_ts(w_inp,beta))
	file_name = dataset + '_ker_list.pkl'
	with open(file_name, 'wb') as f:
		pkl.dump(ker_list,f)
	return ker_list



# DPP sampling for edge
# sample multiple masks for expectations
def dpp_sample_edge_ts(input, weight, beta, k, dataset, num_masks, load_from_pkl = False):

	'''
	return: a list of masks sampled
	[[[inp_dim] * num_masks] * hid_dim]
	'''
	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	# get the kernel based on the inputs and weights
	# one kernel per node (all incoming edges)
	if load_from_pkl:
		file_name = './' + dataset + '_ker_list.pkl'
		ker_list = pkl.load(open(file_name, 'rb'))
		print('loaded kernel', file_name, len(ker_list), ker_list[0].shape)
	else:
		ker_list = create_edge_kernel_ts(input, weight, beta, dataset)
		print('created kernel', str(dataset + '_ker_list.pkl'))

	# [[[inp_dim] * num_masks] * hid_dim]
	samples = []
	for iter_num, ker in enumerate(ker_list):

		# num_masks sampled per kernel (hidden node)
		samples.append(sample_dpp_multiple_ts(ker, k, num_masks))

	# all the masks sampled
	mask_list = [np.zeros((inp_dim, hid_dim)) for _ in range(num_masks)]
	for h_idx in range(len(samples)): # for each hidden node
		for sample_idx, h_sampled in enumerate(samples[h_idx]): # for each sampled kernel
			for k in h_sampled: # for each incoming edge
				mask_list[sample_idx][k][h_idx] = 1

	return mask_list,ker_list 


# DPP sampling for node
# sample multiple masks for expectations
def dpp_sample_node_ts(input, weight, beta, k, num_masks):

	'''
	return: a list of masks sampled
	[[[inp_dim] * num_masks] * hid_dim]
	'''

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	# DPP node kernel
	weighted_input = np.dot(input, weight)
	ker = create_kernel_ts(weighted_input,beta)

	# sample multiple kernels
	# [[hid_dim] * num_masks]
	sample_list = sample_dpp_multiple_ts(ker, k, num_masks)

	# all samples masks
	mask_list = [np.zeros((inp_dim, hid_dim)) for _ in range(num_masks)]
	for num in range(num_masks): 
		for hid_node in sample_list[num]:
			mask_list[num][:, hid_node] = np.ones(inp_dim)
	return mask_list,ker


def reweight_rand_edge(input,weight1,mask,k):

	# weight[edges_in,h] = (1.0/c) * weight[edges_in,h]
	weight = np.copy(weight1)

	num_inp = input.shape[0]
	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]
	c = k / inp_dim
	
	for h in range(hid_dim):
		cur_col = mask[:,h]

		edges_in = np.nonzero(cur_col)[0]
		# edges_not_in = np.where(cur_col == 0)[0]

		# X = input[:,edges_in]
		# y = np.dot(input[:,edges_not_in],weight[edges_not_in,h])

		# assert(X.shape[0]==num_inp and X.shape[1]==edges_in.shape[0] and y.shape[0]==num_inp)

		# clf = LinearRegression(fit_intercept=False)
		# delta = clf.fit(X, y).coef_
		# assert(len(delta)==len(edges_in))
		
		weight[edges_in,h] = (1.0 / c) * weight[edges_in,h]

	return weight



def reweight_edge(input,weight1,mask):

	weight = np.copy(weight1)

	num_inp = input.shape[0]
	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	for h in range(hid_dim):
		cur_col = mask[:,h]

		edges_in = np.nonzero(cur_col)[0]
		edges_not_in = np.where(cur_col == 0)[0]

		X = input[:,edges_in]
		y = np.dot(input[:,edges_not_in],weight[edges_not_in,h])

		assert(X.shape[0]==num_inp and X.shape[1]==edges_in.shape[0] and y.shape[0]==num_inp)

		clf = LinearRegression(fit_intercept=False)
		delta = clf.fit(X, y).coef_
		assert(len(delta)==len(edges_in))
		weight[edges_in,h] = weight[edges_in,h]+delta

	return weight

def reweight_node(input,weight1,weight2,mask):

	num_inp = input.shape[0]
	hid_cur_dim = weight2.shape[0]
	hid_next_dim = weight2.shape[1]

	assert(mask.shape[1]==hid_cur_dim)

	weighted_input = np.dot(input,weight1)

	assert(weighted_input.shape[1]==hid_cur_dim)

	edges_in = np.nonzero(mask[0])[0]
	edges_not_in = np.where(mask[0] == 0)[0]

	alpha_mat = np.zeros((edges_in.shape[0],edges_not_in.shape[0]))
	for i,h in enumerate(edges_not_in):


		X = weighted_input[:,edges_in]
		y = weighted_input[:,h]

		assert(X.shape[0]==num_inp and X.shape[1]==edges_in.shape[0] and y.shape[0]==num_inp)

		clf = LinearRegression(fit_intercept=False)
		alpha = clf.fit(X, y).coef_
		assert(alpha.shape[0]==edges_in.shape[0])
		alpha_mat[:,i] = alpha

	weight2[edges_in,:] += np.dot(alpha_mat, weight2[edges_not_in, :])

	return weight2

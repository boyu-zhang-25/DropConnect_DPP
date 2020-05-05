import numpy as np
from sklearn.metrics import pairwise_distances as pd
import pickle as pkl
from dppy.finite_dpps import FiniteDPP
from sklearn.linear_model import LinearRegression
import torch

def create_kernel(weighted_input,beta):
	#print(weighted_input.shape)
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

def create_edge_kernel(input1, weight1, beta, dataset):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	weight = torch.from_numpy(weight1).float()
	input = torch.from_numpy(input1).float()

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]
	# print('input_shape', input.shape)
	# print('weight_shape', weight.shape)
	ker_list = []
	for h in  range(hid_dim):
		print('creating kernel for hidden node',h)
		w_inp = (weight[:,h])*input
		ker_list.append(create_kernel(w_inp,beta))
	file_name = dataset + '_ker_list.pkl'
	with open(file_name, 'wb') as f:
		pkl.dump(ker_list,f)
	return ker_list




def dpp_sample_edge(input, weight, beta, k, dataset, trained_weights, epsilon=0.01, load_from_pkl = False):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	if load_from_pkl:
		if trained_weights=='mnist_two_layer_Dropout.pt':
			file_name = './' + dataset + '_dropout_ker_list.pkl'
		else:file_name = './' + dataset + '_ker_list.pkl'

		ker_list = pkl.load(open(file_name, 'rb'))
		print('loaded kernel', file_name, len(ker_list), ker_list[0].shape)
	else:
		ker_list = create_edge_kernel(input, weight, beta, dataset)
		print('created kernel', str(dataset + '_ker_list.pkl'))
	samples = []
	for iter_num,ker in  enumerate(ker_list):
		#print(ker.shape)
		ker += epsilon*np.eye(ker.shape[0])
		#print(iter_num,'sampling from DPP')
		samples.append(sample_dpp(ker,k))

	mask = np.zeros((inp_dim,hid_dim))
	for j in range(len(samples)):
		for i in samples[j]:
			mask[i][j] = 1

	return mask


def dpp_sample_node(input,weight,beta,k, trained_weights, epsilon = 0.01, load_from_pkl = False):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	if trained_weights=='mnist_two_layer_Dropout.pt':
		file_name = './' + 'dropout_node_ker.pkl'
	else:file_name = './' + 'node_ker.pkl'
	if load_from_pkl:
		ker = pkl.load(open(file_name, 'rb'))
		print('loaded kernel', file_name, len(ker), ker.shape)
	else:
		weighted_input = np.dot(input,weight)
		ker = create_kernel(weighted_input,beta)
		ker+= epsilon*np.eye(ker.shape[0])
		print('created kernel', file_name)
		with open(file_name, 'wb') as f:
			pkl.dump(ker,f)


	sample = sample_dpp(ker,k)

	mask = np.zeros((inp_dim,hid_dim))
	for hid_node in sample:
		mask[:,hid_node] = np.ones(inp_dim)
	return mask



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

def reweight_node(input,weight1,weight_2,mask):
	weight2 = np.copy(weight_2)
	num_inp = input.shape[0]
	hid_cur_dim = weight2.shape[0]
	hid_next_dim = weight2.shape[1]
	

	assert(mask.shape[1]==hid_cur_dim)

	weighted_input = sigmoid(np.dot(input,weight1))

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
	# fix 
	weight2[edges_not_in,:]=0

	return weight2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
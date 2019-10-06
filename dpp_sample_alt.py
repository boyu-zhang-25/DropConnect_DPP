import numpy as np
from sklearn.metrics import pairwise_distances as pd
import pickle as pkl
from dppy.finite_dpps import FiniteDPP

def alt_create_kernel(weighted_input,beta):
	return np.exp(-beta*(pd(weighted_input.T,metric='l2'))**2)

def alt_sample_dpp(kernel,k):
	DPP = FiniteDPP('likelihood',**{'L':kernel})
	DPP.sample_exact_k_dpp(size=k, mode = 'GS_bis')
	x = list(DPP.list_of_samples)[0]
	assert(len(x)==k)
	return x

# def alt_create_weight(input,weight):
# 	print(input.shape, weight.T.shape, (weight.T)[:,np.newaxis].shape)
# 	return input*(weight.T)[:,np.newaxis]


#input = array of shape (num_inp * inp_dimension)
#weight = array of shape (inp_dim * hid_dim)
#k = the number of incoming edges to keep for each hidden node

def alt_create_edge_kernel(input,weight,beta):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	
	
	ker_list = []
	for h in  range(hid_dim):
		print('creating kernel for hidden node',h)
		w_inp = (weight[:,h])*input
		ker_list.append(alt_create_kernel(w_inp,beta))
	with open('ker_list.pkl','wb') as f:
		pkl.dump(ker_list,f)
	return ker_list
	



def alt_dpp_sample_edge(input,weight,beta,k,load_from_pkl=False):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	if load_from_pkl:
		ker_list = pkl.load(open('ker_list.pkl','rb'))
		print(len(ker_list), ker_list[0].shape)

	else:
		ker_list = alt_create_edge_kernel(input,weight,beta)
	samples = []
	for iter_num,ker in  enumerate(ker_list):
		# print(iter_num,'sampling from DPP')
		samples.append(alt_sample_dpp(ker,k))

	mask = np.zeros((inp_dim,hid_dim))
	for j in range(len(samples)):
		for i in samples[j]:
			mask[i][j] = 1

	return mask


def alt_dpp_sample_node(input,weight,beta,k):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	weighted_input = np.dot(input,weight)
	ker = create_kernel(weighted_input,beta)
	sample = sample_dpp(ker,k)

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

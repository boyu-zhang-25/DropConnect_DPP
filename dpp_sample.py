from dppy.finite_dpps import FiniteDPP 
import numpy as np
from sklearn.metrics import pairwise_distances as pd

def sample_dpp(weighted_input,beta=0.3,k=3):

	ker = np.exp(-beta*(pd(weighted_input,metric='l2'))**2)

	#Constructing DPP with the kernel and sampling
	DPP = FiniteDPP('likelihood',**{'L':ker})
	sample = DPP.sample_exact_k_dpp(size=k)

	return sample


def create_weight(input,weight):

	return input*(weight)[:,np.newaxis])


def mapped_sample(input,weight,beta=0.3,k=3):

	inp_dim = weight.shape[0]
	hid_dim = weight.shape[1]

	weighted_input_mat = create_weight(input,weight)
	samples = []

	for w_inp in weighted_input:

		samples.append(sample_dpp(w_inp))

	mask = np.zeros(inp_dim,hid_dim)	
	for j in range(len(samples)):
		for i in samples[j]:
			mask[i][j] = 1



	return mask


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from random import sample
import math
from torch.utils.data import Dataset
from scipy.special import expit
import numpy as np

# teacher forward call
# gaussian noise 
def teacher_predict(inp, w1, w2, ep):
	h = np.dot(w1, inp) / math.sqrt(inp.shape[0]) # input_dim * 1
	return np.dot(w2, expit(h)) + ep # 1 * 1

class Teacher_dataset(Dataset):
	"""docstring for Teacher_dataset"""
	def __init__(self,
				num_data,
				input_dim,
				teacher_hid_dim,
				sig_w = 1,
				sig_noise = 1):
		super(Teacher_dataset, self).__init__()

		inputs = np.zeros((input_dim, num_data)) # input_dim * num_data
		labels = np.zeros(num_data)

		w1 = np.random.normal(size=(teacher_hid_dim, input_dim)) # teacher_hid_dim * input_dim
		w2 = np.random.normal(size=(1, teacher_hid_dim)) # 1 * teacher_hid_dim 

		for x in range(num_data):

			# single input
			inp = np.random.normal(size = input_dim) # input_dim * 1
			# gaussian noise
			ep = np.random.normal(0, sig_noise)
			# single label
			lab = teacher_predict(inp, w1, w2, ep) # 1 * 1

			inputs[:, x] = inp
			labels[x]= lab

		self.inputs = inputs 
		self.labels = labels

		self.w1 = w1
		self.w2 = w2
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.inputs[idx]

'''
D = Teacher_dataset(num_data = 100,
					input_dim = 10, 
					teacher_hid_dim = 5)

print(D.inputs.shape, D.labels.shape)
'''


		
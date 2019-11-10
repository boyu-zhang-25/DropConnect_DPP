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


def teacher_predict(inp,w1,w2,ep):
	return np.dot(expit(np.dot(inp,w1)),w2)+ep



class Teacher_dataset(Dataset):
	"""docstring for Teacher_dataset"""
	def __init__(self, num_data,input_dim,teacher_hid_dim,sig_w=1,sig_noise=1):
		super(Teacher_dataset, self).__init__()
		inputs = np.zeros((num_data,input_dim))
		labels = np.zeros(num_data)
		w1,w2 = np.random.normal(size=(input_dim,teacher_hid_dim)), np.random.normal(size=(teacher_hid_dim,1))
		for x in range(num_data):
			inp = np.random.normal(size=input_dim)
			ep = np.random.normal(0,sig_noise)
			lab = teacher_predict(inp,w1,w2,ep)
			inputs[x,:] = inp
			labels[x]= lab
		self.inputs = inputs 
		self.labels = labels
		self.w1 = w1
		self.w2 = w2
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.inputs[idx]









# D = Teacher_dataset(1000,5,3)
# print(D.inputs.shape)
		
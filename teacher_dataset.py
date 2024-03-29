from __future__ import print_function
import torch
import math
from torch.utils.data import Dataset
from scipy.special import expit, erf
import numpy as np
import pickle
import argparse

# teacher forward call with gaussian noise 
def teacher_predict(inp, w1, w2, ep, sig_w):

	h = np.dot(w1.data.numpy(), inp.data.numpy()) / math.sqrt(inp.shape[0]) # input_dim * 1
	return np.dot(w2.data.numpy(), erf(h / math.sqrt(2))) + sig_w * ep.data.numpy() # 1 * 1

class Teacher_dataset(Dataset):

	def __init__(self,
				num_data,
				input_dim,
				teacher_hid_dim,
				mode,
				v_star,
				sig_w = 0,
				sig_noise = 1):
		super(Teacher_dataset, self).__init__()

		inputs = torch.zeros((input_dim, num_data)) # input_dim * num_data
		labels = torch.zeros(num_data)

		w1 = torch.randn((teacher_hid_dim, input_dim)) # teacher_hid_dim * input_dim

		if mode == 'soft_committee':
			w2 = torch.ones((1, teacher_hid_dim)) # 1 * teacher_hid_dim 
		else:
			# w2 = np.random.normal(size = (1, teacher_hid_dim)) # 1 * teacher_hid_dim 
			w2 = torch.ones((1, teacher_hid_dim)) * v_star

		# training data
		for x in range(num_data):

			# single input
			inp = torch.randn(input_dim) # input_dim * 1
			# gaussian noise
			ep = torch.randn(1)
			# single label
			lab = teacher_predict(inp, w1, w2, ep, sig_w) # 1 * 1
			# print(type(lab), lab)
			inputs[:, x] = inp
			labels[x]= torch.from_numpy(lab)

		# test data 
		test_num = int(0.1 * num_data)
		test_inputs = torch.zeros((input_dim, test_num)) # input_dim * num_data
		test_labels = torch.zeros(test_num)
		for x_test in range(test_num):
			inp = torch.randn(input_dim) # input_dim * 1
			ep = torch.randn(1)
			lab = teacher_predict(inp, w1, w2, ep, sig_w) # 1 * 1
			test_inputs[:, x_test] = inp
			test_labels[x_test]= torch.from_numpy(lab)			

		# save train and test data
		self.inputs = inputs
		self.labels = labels
		self.test_inputs = test_inputs
		self.test_labels = test_labels

		# save weights
		self.w1 = w1
		self.w1.requires_grad = False
		self.w2 = w2
		self.w2.requires_grad = False		

	def __len__(self):
		return self.inputs.shape[1]

	def __getitem__(self, idx):
		return (self.inputs[:, idx], self.labels[idx])

	def get_test_example(self, idx):
		return (self.test_inputs[:, idx], self.test_labels[idx])


def main():

	parser = argparse.ArgumentParser(description='Teacher-Student Setup')

	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--teacher_h_size', type = int, help='hidden layer size of the student MLP.')
	parser.add_argument('--num_data', type = int, help='Number of data points to be genrated.')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')
	parser.add_argument('--sig_w', type = float, help='scaling variable for the output noise.')
	parser.add_argument('--v_star', type = int, help='ground truth second layer weight')

	# data storage
	parser.add_argument('--teacher_path', type = str, help='Path to store the teacher network and dataset.')
	args = parser.parse_args()

	D = Teacher_dataset(num_data = args.num_data,
						input_dim = args.input_dim, 
						teacher_hid_dim = args.teacher_h_size,
						mode = args.mode,
						v_star = args.v_star, 
						sig_w = args.sig_w)

	print('W2 of the teacher network:', D.w2)
	print('X:', D.inputs.shape, '\nLabels:', D.labels.shape, '\nA random X and its label:', D[0][0].shape, D[0][1].item())
	pickle.dump(D, open(args.teacher_path, "wb"))
	print('Teacher network generated and saved!')

if __name__ == '__main__':
	main()

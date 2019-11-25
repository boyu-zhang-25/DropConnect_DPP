from __future__ import print_function
import torch
import math
from torch.utils.data import Dataset
from scipy.special import expit
import numpy as np
import pickle
import argparse

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
				mode,
				sig_w = 1,
				sig_noise = 1):
		super(Teacher_dataset, self).__init__()

		inputs = np.zeros((input_dim, num_data)) # input_dim * num_data
		labels = np.zeros(num_data)

		w1 = np.random.normal(size = (teacher_hid_dim, input_dim)) # teacher_hid_dim * input_dim

		if mode == 'soft_committee':
			w2 = np.ones((1, teacher_hid_dim)) # 1 * teacher_hid_dim 
		else:
			w2 = np.random.normal(size = (1, teacher_hid_dim)) # 1 * teacher_hid_dim 

		for x in range(num_data):

			# single input
			inp = np.random.normal(size = input_dim) # input_dim * 1
			# gaussian noise
			ep = np.random.normal(0, sig_noise)
			# single label
			lab = teacher_predict(inp, w1, w2, ep) # 1 * 1

			inputs[:, x] = inp
			labels[x]= lab

		# save as troch tensor
		self.inputs = torch.from_numpy(inputs).float()
		self.labels = torch.from_numpy(labels).float()


		self.w1 = torch.from_numpy(w1)
		self.w1.requires_grad = False
		self.w2 = torch.from_numpy(w2)
		self.w2.requires_grad = False

	def __len__(self):
		return self.inputs.shape[1]

	def __getitem__(self, idx):
		return (self.inputs[:, idx], self.labels[idx])


def main():

	parser = argparse.ArgumentParser(description='Teacher-Student Setup')

	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--teacher_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--num_data', type = int, help='Number of data points to be genrated.')
	parser.add_argument('--mode', type = str, help='soft committee machine or two-layer FFNN')

	# data storage
	parser.add_argument('--teacher_path', type = str, help='Path to store the teacher network (dataset).')
	args = parser.parse_args()

	D = Teacher_dataset(num_data = args.num_data,
						input_dim = args.input_dim, 
						teacher_hid_dim = args.teacher_h_size,
						mode = args.mode)

	print('W2 of the teacher network:', D.w2)
	print('X:', D.inputs.shape, '\nLabels:', D.labels.shape, '\nA random X and its label:', D[1][0].shape, D[1][1].item())
	pickle.dump(D, open(args.teacher_path, "wb"))
	print('Teacher network generated and saved!')

if __name__ == '__main__':
	main()

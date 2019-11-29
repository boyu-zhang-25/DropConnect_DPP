from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
from matplotlib import pyplot as plt

from teacher_student import *
from teacher_dataset import *


def get_Q(path_to_mask_list):

	unpurned_MLP, mask_list = pickle.load(open(path_to_mask_list, 'rb'))

	mask_num = len(mask_list)
	w1 = unpurned_MLP.w1.weight.data.cpu().numpy() # hid_dim * inp_dim
	hid_dim, inp_dim = w1.shape[0], w1.shape[1]
	print('student w1 size:', w1.shape, 'mask size:', mask_list[0].T.shape)

	# get the expected Q
	expected_Q = np.zeros((hid_dim, hid_dim))
	for mask in mask_list:
		purned_w = w1 * mask.T
		expected_Q += np.dot(purned_w, purned_w.T)
	expected_Q = expected_Q / mask_num

	# get the unpruned Q
	unpurned_Q = np.dot(w1, w1.T)
	# pickle.dump((expected_Q, unpurned_Q), open('expected_Q', "wb"))

	return expected_Q, unpurned_Q

def plot_Q(expected_Q, unpurned_Q):

	plt.figure(1)
	plt.matshow(expected_Q)
	plt.savefig('expected_Q.png')
	plt.figure(2)
	plt.matshow(unpurned_Q)
	plt.savefig('unpurned_Q.png')


# 
def get_R(path_to_student_mask, path_to_teacher):

	# get the student net
	unpurned_MLP, mask_list = pickle.load(open(path_to_student_mask, 'rb'))
	mask_num = len(mask_list)
	student_w1 = unpurned_MLP.w1.weight.data.cpu().numpy() # student_hid_dim * inp_dim
	student_hid_dim, inp_dim = student_w1.shape[0], student_w1.shape[1]
	print('student w1 size:', student_w1.shape, 'mask size:', mask_list[0].T.shape)

	# get the teacher net
	teacher = pickle.load(open(path_to_teacher, 'rb'))
	teahcer_w1 = teacher.w1.data.cpu().numpy().T # input_dim * teacher_hid_dim
	teacher_hid_dim = teahcer_w1.shape[1]
	print('teacher w1 size:', teahcer_w1.shape)

	# get the expected R on purned student_w1
	# student_hid_dim * teacher_hid_dim
	expected_R = np.zeros((student_hid_dim, teacher_hid_dim))
	for mask in mask_list:
		expected_R += np.dot(student_w1 * mask.T, teahcer_w1)
	expected_R = expected_R / mask_num

	# get the expected R on unpurned student_w1
	unpurned_R = np.dot(student_w1, teahcer_w1)

	# pickle.dump((expected_R, unpurned_R), open('expected_R', "wb"))
	return expected_R, unpurned_R

def plot_R(expected_R, unpurned_R):

	plt.figure(1)
	plt.matshow(expected_R)
	plt.savefig('expected_R.png')
	plt.figure(2)
	plt.matshow(unpurned_R)
	plt.savefig('unpurned_R.png')


def main():

	parser = argparse.ArgumentParser(description='Order Parameter')
	parser.add_argument('--path_to_student_mask', type = str)
	parser.add_argument('--path_to_teacher', type = str, default = 'place_holder')
	args = parser.parse_args()

	expected_Q, unpurned_Q = get_Q(args.path_to_student_mask)
	plot_Q(expected_Q, unpurned_Q)

	expected_R, unpurned_R = get_R(args.path_to_student_mask, args.path_to_teacher)
	plot_R(expected_R, unpurned_R)

if __name__ == '__main__':
	main()

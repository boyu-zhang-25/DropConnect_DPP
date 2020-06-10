from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F 

from teacher_dataset import *
from dpp_sample_ts import *

# the student network
# switch between soft committee machine and two-layer FFNN
class student_MLP(nn.Module):

	def __init__(self,
				input_dim, 
				hidden_size,
				nonlinearity,
				mode,
				device):
		super(student_MLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(input_dim, hidden_size, bias = False) 
		self.w2 = nn.Linear(hidden_size, 1, bias = False) 
		self.input_dim = input_dim

		self.mode = mode
		self.hidden_size = hidden_size

		# choose between ['linear', 'sigmoid', and 'relu']
		if nonlinearity in ['linear', 'sigmoid', 'relu']:
			self.nonlinearity = nonlinearity
		else:
			print('Activation function not supported.')
			raise ValueError

		# nonlinearity layer
		if self.nonlinearity == 'relu':
			self.activation = nn.ReLU()
		elif self.nonlinearity == 'sigmoid':
			# use torch.ERF instead, consistent with [S. Goldt, 2019]
			pass
		else:
			pass			

		self.device = device
		self.initialize()

	# initialization
	def initialize(self):

		if self.nonlinearity == 'sigmoid':
			nn.init.normal_(self.w1.weight.data)
		else:
			nn.init.normal_(self.w1.weight.data, std = 1 / math.sqrt(self.input_dim))

		if self.mode == 'soft_committee':
			print('soft committee machine (freeze student w2 as 1.0)')
			nn.init.ones_(self.w2.weight.data)
			self.w2.weight.requires_grad = False
		else:
			print('two-layer FFNN (DO NOT freeze student w2)')
			if self.nonlinearity == 'sigmoid':
				nn.init.normal_(self.w2.weight.data)
			else:
				nn.init.normal_(self.w2.weight.data, std = 1 / math.sqrt(self.input_dim))
				

	# forward call
	def forward(self, x):

		h = self.w1(x)

		# normalization and ERF activation following [S. Goldt, 2019]
		h_norm = h / math.sqrt(self.input_dim)
		h_norm_new = h_norm / math.sqrt(2)
		a = torch.erf(h_norm_new)
		output = self.w2(a)

		return output


# training loop
# online learning SGD
def train(args, model, device, train_loader, criterion, optimizer, epoch):

	model.train()

	current_error = 0

	# train_loader: input_dim * num_training_data
	for idx in range(len(train_loader)):

		# single data point
		data, target = train_loader[idx]
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)


		# print(output.shape, target.shape)
		loss = criterion(output, target.view(-1))
		loss.backward()

		# manually divide the gradient for auto grad 
		# since Pytorch did not consider 1/2 in the derivative of MSE
		# auto grad does not capture the normalization in [S. Goldt, 2019]
		if model.w2.weight.requires_grad:

			model.w2.weight.grad = model.w2.weight.grad / 2
			model.w1.weight.grad = model.w1.weight.grad * math.sqrt(args.input_dim) / 2

		else:
			model.w1.weight.grad = model.w1.weight.grad * math.sqrt(args.input_dim) / 2

		optimizer.step()
		current_error += loss.item()

		if idx % 100000 == 0 and idx != 0:
			print('Train Example: [{}/{}]\tLoss: {:.6f}\t Epoch[{}]'.format(idx, len(train_loader), current_error, epoch))
			current_error = 0

# testing loop
def test(args, model, device, teacher_dataset, criterion):

	model.eval()
	current_error = 0

	test_num = teacher_dataset.test_inputs.shape[1]
	for idx in range(test_num):

		data, target = teacher_dataset.get_test_example(idx)
		data, target = data.to(device), target.to(device)
		output = model(data)

		loss = criterion(output, target.view(-1))
		current_error += loss.item()

	return current_error / test_num

# get the list for masks for expected kernel
# DOES NOT apply the actual pruning and reweighting
def get_masks(MLP, input, pruning_choice, beta, k, num_masks, device):

	'''
	return:
	unpruned MLP
	a list of masks sampled: [[[inp_dim] * num_masks] * hid_dim]
	'''

	# normalize the weights by L2 
	# print(MLP.w1.weight.shape) # torch.Size([6, 500])
	MLP.w1.weight.data = F.normalize(MLP.w1.weight.data, p = 2, dim  = 1) * np.sqrt(MLP.w1.weight.shape[1])

	# input_dim * hidden_size
	original_w1 = MLP.w1.weight.data.cpu().numpy().T # [500, 6]
	print('original_w1 size', original_w1.shape)

	# num_training_data * input_dim
	input = input.cpu().numpy()
	print('input size', input.shape)

	mask_list = None

	if pruning_choice == 'dpp_edge':

		# input_dim * hidden_size
		mask_list, ker = dpp_sample_edge_ts(
								input = input, 
								weight = original_w1, 
								beta = beta, 
								k = k, 
								dataset = 'student_' + str(original_w1.shape[1]) + '_w1_'+str(k),
								num_masks = num_masks,
								load_from_pkl = False)

		print('dpp_edge mask_list length:', len(mask_list), 'each mask shape:', mask_list[0].shape)

	elif pruning_choice == 'dpp_node':
		mask_list, ker  = dpp_sample_node_ts(
								input = input, 
								weight = original_w1, 
								beta = beta, 
								k = k, 
								num_masks = num_masks)

		print('dpp_node mask_list length:', len(mask_list), 'each mask shape:', mask_list[0].shape)

	elif pruning_choice == 'random_edge':

		prob = float(k) / MLP.input_dim
		mask_list = [np.random.binomial(1, prob, size=original_w1.shape) for _ in range(num_masks)]
		print('random mask_list length:', len(mask_list), 'each mask shape:', mask_list[0].shape)
		ker = []

	elif pruning_choice == 'importance_edge':
		mask = np.ones(original_w1.shape)
		k = original_w1.shape[0] - k
		for h in range(original_w1.shape[1]):
			onorm_idx = np.argpartition(np.squeeze(np.abs(original_w1)[:, h]), k)[:k]
			print('onorm_idx shape:', onorm_idx.shape)
			mask[:, h][onorm_idx] = 0

		mask_list = [mask]
		ker = []

	elif pruning_choice == 'importance_node':
		mask = np.ones(original_w1.shape)
		original_w2 = MLP.w2.weight.data.cpu().numpy() # [1, 6]
		# onorm = np.sum(np.abs(original_w1), axis = 0)
		k = original_w1.shape[1] - k
		onorm_idx = np.argpartition(np.squeeze(original_w2), k)[:k]
		print('onorm_idx', onorm_idx.shape)
		for h in onorm_idx:
			mask[:, h] = 0

		mask_list = [mask]
		ker = []

	elif pruning_choice == 'random_node':

		mask_list = []
		for num in range(num_masks):

			mask = np.ones(original_w1.shape)
			idx = np.zeros(original_w1.shape[1])
			idx[:k] = 1
			np.random.shuffle(idx)
			for i, val in enumerate(idx):
				mask[:, i] = val

			mask_list.append(mask)

		ker = []

	else:
		print('pruning method not defined in Teacher-Student setup!')
		raise ValueError

	return MLP, mask_list, ker


# def plot_rwt_R(expected_R):

# 	plt.figure(1)
# 	fig, ax = plt.subplots()
# 	expected_R = abs(expected_R)
# 	im = ax.imshow(expected_R)
# 	plt.colorbar(im);
# 	fig.tight_layout()
# 	plt.savefig('expected_rwt_R.png', dpi = 100)



# def plot_rwt_Q(expected_Q):

# 	plt.figure(1)
# 	fig, ax = plt.subplots()
# 	expected_Q = abs(expected_Q)
# 	im = ax.imshow(expected_Q)
# 	plt.colorbar(im);
# 	fig.tight_layout()
# 	plt.savefig('expected_rwt_Q.png', dpi = 100)




def main():

	# hyperparameter settings
	parser = argparse.ArgumentParser(description='Teacher-Student Setup')

	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--nonlinearity', type = str, help='choice of the activation function')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')

	# optimization setup
	parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
						help='learning rate (default: 0.5) that will be scaled for each layer accordingly')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0)')
	parser.add_argument('--epoch', type = int, default = 1, help='number of epochs (online learning so 1)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# pruning parameters
	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, dpp_node, random_node, importance_edge, importance_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 2,
						help='number of parameters to preserve (for node pruning: # of nodes; for edge pruning: # of weights per node)')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training, pruning, or testing')
	parser.add_argument('--num_masks', type = int, default = 1,
						help='Number of masks to be sampled by DPP.')
	parser.add_argument('--reweighting', type = int, default = 0,
						help='if to reweight the pruned network.')

	# data storage
	parser.add_argument('--trained_weights', type = str, default = 'place_holder', help='path to the trained weights for loading')
	parser.add_argument('--teacher_path', type = str, help='Path to store the teacher network and dataset.')
	args = parser.parse_args()

	# print(args)
	# CUDA
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('device', device)

	# reproducibility
	torch.manual_seed(args.seed)

	# create the model, loss, and optimizer
	model = student_MLP(input_dim = args.input_dim,
						hidden_size = args.student_h_size,
						nonlinearity = args.nonlinearity,
						mode = args.mode,
						device = device).to(device)

	# soft committee machine; freeze the second layer
	if args.mode == 'soft_committee':
		print('soft_committee!')
		model.w2.weight.requires_grad = False
		optimizer = optim.SGD([
								{'params': [model.w1.weight], 'lr' : args.lr / np.sqrt(args.input_dim)},
								], lr = args.lr, momentum = args.momentum)
	else:
		print('two-layers NN!')

		# scale the LR for differnent layers
		optimizer = optim.SGD([
								{'params': [model.w1.weight], 'lr' : args.lr / np.sqrt(args.input_dim)},
								{'params': [model.w2.weight], 'lr' : args.lr / args.input_dim}
								], lr = args.lr, momentum = args.momentum)
	criterion = nn.MSELoss()

	if torch.cuda.device_count() > 1 and args.procedure == 'training':
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# model = nn.DataParallel(model)

	# get the data set from the teacher network
	# it is just the teacher dataset... bad naming :)
	train_loader = pickle.load(open(args.teacher_path, "rb" ))

	# train 
	if args.procedure == 'training':
		print('Training started!')

		# online SGD
		for epoch in range(args.epoch):
			train(args, model, device, train_loader, criterion, optimizer, epoch)

		torch.save(model.state_dict(), 'student_' + str(args.student_h_size) + '.pth')


	# pruning
	elif args.procedure == 'pruning':
		print('Pruning started!')
		model.eval()

		# inference only
		with torch.no_grad():

			# load the model every iteration
			model.load_state_dict(torch.load(args.trained_weights, map_location = torch.device('cpu')))

			# sampled masks 
			unpruned_MLP, mask_list, ker = get_masks(
												MLP = model, 
												input = train_loader.inputs.T, 
												pruning_choice = args.pruning_choice, 
												beta = args.beta, 
												k = args.k, 
												num_masks = args.num_masks,
												device = device)

			# if args.pruning_choice == 'dpp_node':

			# 	ker = (1.0 / args.input_dim) * abs(ker)
			# 	plt.figure()
			# 	fig, ax = plt.subplots()
			# 	im = plt.imshow(ker)

			# 	# for i in range(len(ker)):
			# 	# 	for j in range(len(ker)):
			# 	# 		text = ax.text(j, i, '%.3f' % ker[i, j],
			# 	# 					   ha = "center", va = "center", color = "w")

			# 	plt.colorbar(im)
			# 	plt.tight_layout()
			# 	plt.savefig("theoretical_node_kernel.png", dpi = 200)
			# 	plt.close()

			# if args.pruning_choice == 'dpp_edge':

			# 	ker = (1.0 / args.input_dim) * abs(np.array(ker))
			# 	print("Shape of Kernel List: ",ker.shape)
			# 	for ind,k in enumerate(ker):
			# 		print("Sum of diagonals: ", sum([k[i][i] for i in range(args.input_dim)]))
	
			# 		plt.figure()
			# 		fig, ax = plt.subplots()
			# 		im = plt.imshow(k)

			# 		# for i in range(len(k)):
			# 		# 	for j in range(len(k)):
			# 		# 		text = ax.text(j, i, '%.3f' % k[i, j],
			# 		# 					   ha = "center", va = "center", color = "w")

			# 		plt.colorbar(im)
			# 		plt.tight_layout()
			# 		plt.savefig("theoretical_edge_kernel_node_" + str(ind) + ".png", dpi = 200)
			# 		plt.close()

			file_name = 'student_masks_' + args.pruning_choice + '_' + str(args.student_h_size)+'_' + str(args.k) + '.pkl'
			pickle.dump((unpruned_MLP, mask_list), open(file_name, "wb"))

	# testing
	else:
		print('Testing:', args.pruning_choice, '\nstudent hidden size:', args.student_h_size, '\nk:', args.k)
		print('rwt:', args.reweighting)

		# inference only
		with torch.no_grad():

			# load the unpruned model and masks
			file_name = 'student_masks_' + args.pruning_choice + '_' + str(args.student_h_size) + "_" + str(args.k) + '.pkl'
			unpruned_MLP, mask_list = pickle.load(open(file_name, 'rb'))
			
			# print('norm:', unpruned_MLP.w1.weight.data.norm(p = 2, dim  = 1, keepdim = True))

			# test_loss = test(args, unpruned_MLP, device, train_loader, criterion)
			# print('unpruned MLP avg. test loss:', test_loss)


			# apply DPP pruning and get the the expected test loss
			old_w1 = unpruned_MLP.w1.weight.data
			old_w2 = unpruned_MLP.w2.weight.data
			pruned_test_loss = 0


			# expected_rwt_Q = np.zeros((args.student_h_size, args.student_h_size))
			# teahcer_w1 = train_loader.w1.data.cpu().numpy().T # input_dim * teacher_hid_dim
			# expected_rwt_R = np.zeros((args.student_h_size, teahcer_w1.shape[1]))

			test_num = 5
			for mask_idx, mask in enumerate(mask_list[:test_num]):

				# print(mask.shape, original_w1.shape, )
				original_w1 = np.copy(old_w1.cpu().numpy())
				original_w2 = np.copy(old_w2.cpu().numpy().T)
				# pruned_w1 = torch.from_numpy(original_w1 * mask.T)
				# unpruned_MLP.w1.weight.data = pruned_w1.float().to(device)
				unpruned_MLP.w1.weight.data = torch.from_numpy(original_w1 * mask.T).float().to(device)

				# apply reweighting
				if args.reweighting:

					if args.pruning_choice.endswith('_node'):

						reweighted_w2 = reweight_node(train_loader.inputs.T, original_w1.T, original_w2, mask)
						unpruned_MLP.w2.weight.data = torch.from_numpy(reweighted_w2.T).float().to(device)
						# print('applied reweighting')

					elif args.pruning_choice.endswith('_edge'):

						# print('apply reweight_rand_edge')
						# reweighted_w1 = reweight_rand_edge(train_loader.inputs.T, original_w1.T, mask, args.k)
						reweighted_w1 = reweight_edge(train_loader.inputs.T, original_w1.T, mask)

						# pruned_w1 = torch.from_numpy((mask * reweighted_w1).T)
						# unpruned_MLP.w1.weight.data = pruned_w1.float().to(device)
						unpruned_MLP.w1.weight.data = torch.from_numpy((mask * reweighted_w1).T).float().to(device)

						# expected_rwt_Q += np.dot(pruned_w1, pruned_w1.T)
						# expected_rwt_R += np.dot(pruned_w1, teahcer_w1)

					else:
						print('pruning method not defined and not re-weighting avaliable!')
						raise ValueError

				# unpruned_MLP.w1.weight.data *= torch.from_numpy(mask.T)
				pruned_test_loss += test(args, unpruned_MLP, device, train_loader, criterion)
				# print(pruned_test_loss)
				# unpruned_MLP.w1.weight.data = old_w1

			# expected_rwt_Q = expected_rwt_Q / len(mask_list)
			# expected_rwt_Q = expected_rwt_Q / teahcer_w1.shape[0]
			# expected_rwt_R = expected_rwt_R / len(mask_list)
			# expected_rwt_R = expected_rwt_R / teahcer_w1.shape[0]


			# plot_rwt_Q(expected_rwt_Q)
			# plot_rwt_R(expected_rwt_R)

				# if mask_idx % 20 == 0 and mask_idx != 0:
				# 	print('Tested Masks: [{}/{}]\tAvg. Loss: {:.6f}\t'.format(mask_idx, len(mask_list), pruned_test_loss / mask_idx / 2.0))

			if test_num >= len(mask_list):
				print('pruned MLP avg. test loss:', pruned_test_loss / len(mask_list) / 2)
			else:
				print('pruned MLP avg. test loss:', pruned_test_loss / test_num / 2)				


if __name__ == '__main__':
	main()

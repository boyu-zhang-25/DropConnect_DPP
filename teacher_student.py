from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import math
from dpp_sample import *
from teacher_dataset import *
import pickle

# the student network
# switch between soft committee machine and two-layer FFNN
class student_MLP(nn.Module):

	def __init__(self,
				input_dim, 
				hidden_size,
				nonlinearity,
				device):
		super(MLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(input_dim, hidden_size) 
		self.w2 = nn.Linear(hidden_size, 1) 

		self.hidden_size = hidden_size

		# choose between ['linear', 'sigmoid', and 'relu']
		if nonlinearity in ['linear', 'sigmoid', and 'relu']:
			self.nonlinearity = nonlinearity
		else:
			print('Activation function not supported.')
			raise ValueError

		# nonlinearity layer
		if self.nonlinearity == 'relu':
			self.activation = nn.ReLU()
		elif self.nonlinearity == 'sigmoid':
			self.activation = nn.Sigmoid()
		else:
			pass			

		self.device = device
		self.initialize()

	# Xavier initialization
	def initialize(self):

		nn.init.xavier_uniform_(self.w1.weight.data, 
								gain = nn.init.calculate_gain(self.nonlinearity))
		nn.init.xavier_uniform_(self.w2.weight.data, 
								gain = nn.init.calculate_gain(self.nonlinearity))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		
	# forward call
	def forward(self, x):
		x = self.activation(self.w1(x))
		return self.w2(x)


# training loop
# online learning SGD
def train(args, model, device, train_loader, criterion, optimizer, epoch):

	model.train()

	# train_loader: input_dim * num_training_data
	for idx in range(len(train_loader)):

		# single data point
		data, target = train_loader[idx]
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		# print(output.shape, target.shape)
		loss = criterion(output, target)

		loss.backward()
		optimizer.step()

		if idx % 100 == 0:
			print('Train Example: [{}/{}]\tLoss: {:.6f}'.format(idx, len(train_loader), loss.item()))


# apply post pruning on the two layer MLP and test
# for w1 in the student network
def prune_MLP_w1(MLP, input, pruning_choice, reweighting, beta, k, device):

	# input_dim * hidden_size
	dpp_weight = 0
	original_w1 = MLP.w1.weight.data.cpu().numpy().T
	print('original_w1 size', original_w1.shape)

	# num_training_data * input_dim
	input = input.cpu().numpy()
	print('input size', input.shape)

	mask = None

	if pruning_choice == 'dpp_edge':

		# input_dim * hidden_size
		mask = dpp_sample_edge(input, original_w1, beta = beta, k = k, dataset = 'student_' + original_w1.shape[1] + '_w1')
		if reweighting:
			dpp_weight = reweight_edge(input,original_w1,mask)
		print('dpp_edge mask size', mask.shape)

	elif pruning_choice == 'dpp_node':
		mask = dpp_sample_node(input, original_w1, beta = beta, k = k)
		print('dpp_node mask size', mask.shape)
	elif pruning_choice == 'random_edge':
		mask = np.random.binomial(1,0.5,size=original_w1.shape)

	pruned_w1 = torch.from_numpy((mask * original_w1).T)
	print('pruned_w1 size', pruned_w1.shape)

	with torch.no_grad():
		MLP.w1.weight.data = pruned_w1.float().to(device)

	return MLP, dpp_weight, mask



def main():

	# hyperparameter settings
	parser = argparse.ArgumentParser(description='Teacher-Student Setup')

	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--nonlinearity', type = str, help='choice of the activation function')
	parser.add_argument('--mode', type = str, help='soft committee machine or two-layer FFNN')

	# optimization setup
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# pruning parameters
	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, dpp_node, random_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 20,
						help='number of edges/nodes to preserve')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training or purning')
	parser.add_argument('--reweighting', action='store_true', default = False,
						help='For fusing the lost information')

	# data storage
	parser.add_argument('--trained_weights', type = str, help='path to the trained weights for loading')
	parser.add_argument('--teacher_path', type = str, help='Path to the teacher network (dataset).')
	args = parser.parse_args()

	# print(args)
	# CUDA
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# reproducibility
	torch.manual_seed(args.seed)

	# create the model, loss, and optimizer
	model = student_MLP(input_dim = args.input_dim,
						hidden_size = args.student_h_size,
						nonlinearity = args.nonlinearity,
						device = device).to(device)

	# soft committee machine; freeze the second layer
	if args.mode == 'soft_committee':
		model.w2.requires_grad = False
		optimizer = optim.SGD(model.w1,
								lr = args.lr,
								momentum = args.momentum)
	else:
		optimizer = optim.SGD(model.parameters(),
								lr = args.lr,
								momentum = args.momentum)		
	criterion = nn.MSELoss()

	if torch.cuda.device_count() > 1 and args.procedure == 'training':
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# model = nn.DataParallel(model)

	# get the data set from the teacher network
	train_loader = pickle.load(open(args.teacher_path, "rb" ))

	# train 
	if args.procedure == 'training':

		# online SGD
		for epoch in range(1):
			train(args, model, device, train_loader, criterion, optimizer, epoch)
		torch.save(model.state_dict(), 'student_' + args.student_h_size + '.pth')

	# pruning
	else:

		model.eval()

		# inference only
		with torch.no_grad():

			# load the model every iteration
			model.load_state_dict(torch.load(args.trained_weights, map_location = torch.device('cpu')))

			# faltten the image
			test_all_data = test_all_data.view(test_all_data.shape[0], -1)
			train_all_data = train_all_data.view(train_all_data.shape[0], -1)
			test_all_data, target = test_all_data.to(device), target.to(device)
			train_all_data = train_all_data.to(device)

			# get the processed hidden layer as input for pruning w2
			# batch_size * hidden_size
			hidden_tensors = model.relu(model.w1(train_all_data))

			# prune the w1
			model, dpp_weight_w1, mask_w1 = prune_MLP_w1(model, train_all_data, args.pruning_choice, args.reweighting, args.beta, args.k, device = device)
			file_name = 'pruned_student_' + args.student_h_size + '.pkl'
			pickle.dump((model, dpp_weight_w1, mask_w1), open(file_name, "wb"))


if __name__ == '__main__':
	main()

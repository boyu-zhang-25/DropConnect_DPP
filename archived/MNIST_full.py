from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from random import sample
import math
from dpp_sample import *

# full size MLP (same as the DIVNET)
# 784-500-500-10
class MLP(nn.Module):

	def __init__(self,
				hidden_size,
				drop_option,
				probability,
				device):
		super(MLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(28 * 28, hidden_size) # hidden_size * 784
		self.hidden_size = hidden_size
		self.relu = nn.ReLU()
		self.drop_option = drop_option
		self.probability = probability
		self.w2 = nn.Linear(hidden_size, hidden_size) # hidden_size * hidden_size
		self.w3 = nn.Linear(hidden_size, 10) # 10 * hidden_size
		self.device = device
		self.initialize()

		# the drop layer
		if drop_option == 'out':
			print('Using Dropout with p = {}'.format(probability))
			self.dropout = nn.Dropout(p = probability)
		elif drop_option == 'connect':
			print('Using DropConnect with p = {}'.format(probability))


	# Xavier init
	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w2.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w3.weight.data, gain = nn.init.calculate_gain('relu'))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		self.w3.bias.data.zero_()

	def forward(self, x):

		# batch_size * 784 -> batch_size * hidden_size
		# switch between dropout/DropConnect
		if self.drop_option == 'out':
			x = self.relu(self.w1(x))
			x = self.dropout(x)
		elif self.drop_option == 'connect':

			# only apply during training
			if self.training:
				x = self.drop_connect(x, layer_choice = 'w1')
				x = self.relu(x)
			else:
				x = self.relu(self.w1(x))
		else:
			x = self.relu(self.w1(x))

		# batch_size * hidden_size -> batch_size * hidden_size
		x = self.relu(self.w2(x))

		# batch_size * hidden_size -> batch_size * 10
		x = self.w3(x)

		return x

	# drop connect on w1
	# different masks for each example in the same batch
	def drop_connect(self, x, layer_choice):

		# [batch_size, hidden_size]
		result = torch.zeros(x.shape[0], self.hidden_size).to(self.device)

		# [hidden_size, 784, batch_size]
		mask = torch.bernoulli(self.probability * torch.ones(
															self.w1.weight.shape[0],
															self.w1.weight.shape[1],
															x.shape[0])).to(self.device)

		# TODO: vectorization
		# for each example in the batch
		for batch in range(x.shape[0]):

			old_weight = self.w1.weight.data
			# mask out connections for each example
			# self.w1.weight.data.mul_(mask[:, :, batch])
			self.w1.weight.data = self.w1.weight * mask[:, :, batch]
			result[batch] = self.w1(x[batch])
			self.w1.weight.data = old_weight

		#  print(result.grad_fn)
		return result

# training loop
def train(args, model, device, train_loader, criterion, optimizer, epoch):
	model.train()
	correct = 0
	for batch_idx, (data, target) in enumerate(train_loader):

		# faltten the image
		# torch.Size([64, 1, 28, 28]) -> [64, 28*28]
		# torch.Size([64])
		data = data.view(data.shape[0], -1)
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		# print(output.shape, target.shape)
		loss = criterion(output, target)

		pred = output.argmax(dim = 1, keepdim = True)
		correct += pred.eq(target.view_as(pred)).sum().item()

		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))


	print('train Accuracy is: ', 100. * correct / len(train_loader.dataset))

# testing loop
def test(args, model, device, test_loader, criterion):
	model.eval()
	test_loss = 0
	correct = 0

	# inference only
	with torch.no_grad():
		for data, target in test_loader:

			# faltten the image
			# torch.Size([64, 1, 28, 28]) -> [64, 28*28]
			# torch.Size([64])
			data = data.view(data.shape[0], -1)
			data, target = data.to(device), target.to(device)
			output = model(data)

			# sum up batch loss
			test_loss += criterion(output, target).item()

			# get the index of the max log-probability
			pred = output.argmax(dim = 1, keepdim = True)
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

# apply post pruning on the two layer MLP and test
# for w1 in the full model
def prune_MLP_w1(MLP, input, pruning_choice, reweighting, beta, k, device):

	# 784 * hidden_size
	dpp_weight = 0
	original_w1 = MLP.w1.weight.data.cpu().numpy().T
	print('w1', original_w1.shape)

	# batch_size * 784
	input = input.cpu().numpy()
	print('input', input.shape)

	mask = None

	if pruning_choice == 'dpp_edge':

		# 784 * hidden_size
		mask = dpp_sample_edge(input, original_w1, beta = beta, k = k, dataset = 'MNIST_full_w1')

		if reweighting:
			dpp_weight = reweight_edge(input,original_w1,mask)

		print('mask', mask.shape)

	elif pruning_choice == 'dpp_node':
		mask = dpp_sample_node(input, original_w1, beta = beta, k = k)

	elif pruning_choice == 'random_edge':
		mask = np.random.binomial(1,0.5,size=original_w1.shape)

	pruned_w1 = torch.from_numpy((mask * original_w1).T)
	print('pruned_w1', pruned_w1.shape)

	with torch.no_grad():
		MLP.w1.weight.data = pruned_w1.float().to(device)

	return MLP, dpp_weight, mask

# apply post pruning on the two layer MLP and test
# for w2 in the full model
# input should be processed by w1 first
def prune_MLP_w2(MLP, input, pruning_choice, reweighting, beta, k, device):

	# 784 * hidden_size
	dpp_weight = 0
	original_w2 = MLP.w2.weight.data.cpu().numpy().T
	print('w2', original_w2.shape)

	# batch_size * 784
	input = input.cpu().numpy()
	print('input', input.shape)

	mask = None
	if pruning_choice == 'dpp_edge':
		# 784 * hidden_size
		mask = dpp_sample_edge(input, original_w2, beta = beta, k = k, dataset = 'MNIST_full_w2')
		if reweighting:
			dpp_weight = reweight(input,original_w2,mask)
		print('mask', mask.shape)

	elif pruning_choice == 'dpp_node':
		mask = dpp_sample_node(input, original_w2, beta = beta, k = k)

	elif pruning_choice == 'random_edge':
		mask = np.random.binomial(1,0.5,size=original_w2.shape)

	pruned_w2 = torch.from_numpy((mask * original_w2).T)
	print('pruned_w2', pruned_w2.shape)

	with torch.no_grad():
		MLP.w2.weight.data = pruned_w2.float().to(device)

	return MLP, dpp_weight, mask

def main():

	# hyperparameter settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
						help='input batch size for training (default: 1000)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type = int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=True,
						help='For Saving the current Model')
	parser.add_argument('--drop-option', type = str,
						help='out or connect')
	parser.add_argument('--probability', type = float,
						help='probability for dropping')
	parser.add_argument('--hidden-size', type = int, default = 500,
						help='hidden layer size of the two-layer MLP')
	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, dpp_node, random_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 20,
						help='number of edges/nodes to preserve')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training or purning')
	parser.add_argument('--trained_weights', type = str, default = 'MNIST_full.pt',
						help='path to the trained weights for loading')
	parser.add_argument('--reweighting', action='store_true', default = False,
						help='For fusing the lost information')
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
	model = MLP(hidden_size = args.hidden_size,
							drop_option = args.drop_option,
							probability = args.probability,
							device = device).to(device)
	optimizer = optim.SGD(model.parameters(),
							lr = args.lr,
							momentum = args.momentum)
	criterion = nn.CrossEntropyLoss()

	if torch.cuda.device_count() > 1 and args.procedure == 'training':
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		# model = nn.DataParallel(model)

	# traning
	if args.procedure == 'training':

		# training data
		train_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train = True, download = True,
						   transform = transforms.Compose([
							   transforms.ToTensor(),

							   # the mean and std of the MNIST dataset
							   transforms.Normalize((0.1307,), (0.3081,))
						   ])),
			batch_size = args.batch_size, shuffle=False, **kwargs)

		# testing data
		test_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train = False, transform = transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Normalize((0.1307,), (0.3081,))
						   ])),
			batch_size = args.test_batch_size, shuffle = False, **kwargs)

		for epoch in range(1, args.epochs + 1):
			train(args, model, device, train_loader, criterion, optimizer, epoch)
			test(args, model, device, test_loader, criterion)

		if (args.save_model):
			if args.drop_option == 'out':
				torch.save(model.state_dict(),"MNIST_full_Dropout.pt")
			elif args.drop_option == 'connect':
				torch.save(model.state_dict(),"MNIST_full_DropConnect.pt")
			else:
				torch.save(model.state_dict(),"MNIST_full.pt")

	# pruning and testing
	else:

		# calculate DPP by all training examples
		train_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train = True, download = True,
						   transform = transforms.Compose([
							   transforms.ToTensor(),

							   # the mean and std of the MNIST dataset
							   transforms.Normalize((0.1307,), (0.3081,))
						   ])),
			batch_size = 60000, shuffle=False, **kwargs)
		train_whole_batch = enumerate(train_loader)
		dummy_idx, (train_all_data, dummy_target) = next(train_whole_batch)
		#print(train_all_data.shape, dummy_target.shape)

		# test on all test data at once
		test_loader = torch.utils.data.DataLoader(
			datasets.MNIST('../data', train = False, transform = transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Normalize((0.1307,), (0.3081,))
						   ])),
			batch_size = 10000, shuffle = False, **kwargs)
		test_whole_batch = enumerate(test_loader)
		dummy_idx, (test_all_data, target) = next(test_whole_batch)
		# print(test_all_data.shape, target.shape)

		model.eval()
		test_loss = 0
		correct = 0
		reweight_test_loss = 0
		reweight_correct = 0

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

			# prune the w2
			model, dpp_weight_w2, mask_w2 = prune_MLP_w2(model, hidden_tensors, args.pruning_choice, args.reweighting, args.beta, args.k, device = device)

			output = model(test_all_data)

			# sum up batch loss
			test_loss += criterion(output, target).item()

			# get the index of the max log-probability
			pred = output.argmax(dim = 1, keepdim = True)
			correct += pred.eq(target.view_as(pred)).sum().item()

			if args.reweighting:

				pruned_w1 = torch.from_numpy((mask * dpp_weight).T)
				model.w1.weight.data = pruned_w1.float().to(device)

				reweight_output = model(test_all_data)

				# sum up batch loss
				reweight_test_loss += criterion(reweight_output, target).item()

				# get the index of the max log-probability
				reweight_pred = reweight_output.argmax(dim = 1, keepdim = True)
				reweight_correct += reweight_pred.eq(target.view_as(reweight_pred)).sum().item()


		test_loss /= len(test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))

		if args.reweighting:
			reweight_test_loss /= len(test_loader.dataset)
			print('\nTest set: Average reweight loss: {:.4f}, Reweighting Accuracy : {}/{} ({:.0f}%)\n'.format(
			reweight_test_loss, reweight_correct, len(test_loader.dataset),
			100. * reweight_correct / len(test_loader.dataset)))


if __name__ == '__main__':
	main()

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from random import sample
import math
from dpp_sample import *

# MLP of size 3072 - 1000 - 1000 - 1000 - 10 (DIVNET, ICLR 2016)
class MLP(nn.Module):

	def __init__(self,
				probability, 
				device,
				activation = 'sigmoid',
				hidden_size = 1000):
		super(MLP, self).__init__()

		# DIVNET
		self.w1 = nn.Linear(32 * 32 * 3, hidden_size) # hidden_size * (32 * 32 * 3)
		self.w2 = nn.Linear(hidden_size, 1000) 
		self.w3 = nn.Linear(1000, 1000) 
		self.w4 = nn.Linear(1000, 10) 

		self.hidden_size = hidden_size
		self.activation_choice = activation

		if activation == 'sigmoid':
			self.activation = nn.Sigmoid()
		else:
			self.activation = nn.ReLU()

		self.probability = probability
		self.device = device
		self.initialize()

		print('Using Dropout with p = {}'.format(probability))
		self.dropout = nn.Dropout(p = probability)


	# Xavier init
	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, 
			gain = nn.init.calculate_gain(self.activation_choice))
		nn.init.xavier_uniform_(self.w2.weight.data, 
			gain = nn.init.calculate_gain(self.activation_choice))
		nn.init.xavier_uniform_(self.w3.weight.data, 
			gain = nn.init.calculate_gain(self.activation_choice))
		nn.init.xavier_uniform_(self.w4.weight.data, 
			gain = nn.init.calculate_gain(self.activation_choice))

		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		self.w3.bias.data.zero_()
		self.w4.bias.data.zero_()


	def forward(self, x):


		# batch_size * (32 * 32 * 3) -> batch_size * hidden_size
		x = self.dropout(self.activation(self.w1(x)))
		x = self.dropout(self.activation(self.w2(x)))
		x = self.dropout(self.activation(self.w3(x)))

		# batch_size * 1000 -> batch_size * 10
		x = self.w4(x)

		# print(self.w1.weight.grad_fn, self.w2.weight.grad_fn, x.requires_grad)
		# batch_size * 10
		return x


# training loop
def train(args, model, device, train_loader, criterion, optimizer, epoch):
	model.train()
	correct = 0
	for batch_idx, (data, target) in enumerate(train_loader):

		# faltten the image
		# torch.Size([batch_size, 3, 32, 32]) -> [batch_size, 3*32*32]
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

	train_acc = 100. * correct / len(train_loader.dataset)
	print('Train Accuracy is: {}%'.format(train_acc))
	return train_acc



# testing loop
def test(args, model, device, test_loader, criterion):
	model.eval()
	test_loss = 0
	correct = 0

	# inference only
	with torch.no_grad():
		for data, target in test_loader:

			# faltten the image
			data = data.view(data.shape[0], -1)
			data, target = data.to(device), target.to(device)
			output = model(data)

			# sum up batch loss
			test_loss += criterion(output, target).item()

			# get the index of the max log-probability
			pred = output.argmax(dim = 1, keepdim = True)
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss = test_loss / len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))



# apply DPP pruning on DIVNET for CIFAR
def prune_MLP(MLP, input, pruning_choice, reweighting, beta, k, trained_weights, device):

	# (32*32*3) * hidden_size
	dpp_weight = None
	original_w1 = MLP.w1.weight.data.cpu().numpy().T
	original_w2 = MLP.w2.weight.data.cpu().numpy().T
	print('w1 shape:', original_w1.shape)

	# batch_size * (3*32*32)
	input = input.numpy()
	print('input shape:', input.shape)

	mask = None

	if pruning_choice == 'dpp_edge':

		# (3*32*32) * hidden_size
		mask = dpp_sample_edge(input = input, 
								weight = original_w1, 
								beta = beta, 
								k = k, 
								trained_weights = trained_weights, 
								load_from_pkl = True)


		if reweighting:
			dpp_weight = reweight_edge(input, original_w1, mask)
		print('mask shape:', mask.shape)

	elif pruning_choice == 'dpp_node':

		mask = dpp_sample_node(input = input, 
								weight = original_w1, 
								beta = beta, 
								k = k, 
								trained_weights = trained_weights,
								load_from_pkl = True)

		if reweighting:
			dpp_weight2 = reweight_node(input,original_w1,original_w2,mask)
			reweighted_w2 = dpp_weight2.T
			#print(reweighted_w2[:,1])
			with torch.no_grad():
				MLP.w2.weight.data = torch.from_numpy(reweighted_w2).float().to(device)

	elif pruning_choice == 'random_edge':
		prob = float(k) / 3072
		mask = np.random.binomial(1, prob, size = original_w1.shape)

	# apply the mask
	pruned_w1 = torch.from_numpy((mask * original_w1).T)
	print('pruned_w1 shape:', pruned_w1.shape)

	with torch.no_grad():
		MLP.w1.weight.data = pruned_w1.float().to(device)

	return MLP, dpp_weight, mask



def main():

	# hyperparameter settings
	parser = argparse.ArgumentParser(description='DPP on CIFAR 10')
	parser.add_argument('--train_batch_size', type=int, default=1000, metavar='N',
						help='input batch size for training (default: 1000)')
	parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')

	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.001)')
	parser.add_argument('--momentum', type=float, default = 0.9, metavar='M',
						help='SGD momentum (default: 0.9)')

	parser.add_argument('--no_cuda', action='store_true', default=False,
						help='disables CUDA training')

	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	parser.add_argument('--log_interval', type = int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save_model', action='store_true', default=True,
						help='For Saving the current Model')

	parser.add_argument('--probability', type = float, default = 0.5, 
						help='probability for dropout')

	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, or dpp_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 20,
						help='number of edges/nodes to preserve')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training or purning')
	parser.add_argument('--trained_weights', type = str, default = 'CIFAR_0.0_batch128.pth',
						help='path to the trained weights for loading')
	parser.add_argument('--reweighting', action='store_true', default = False,
						help='For fusing the lost information')
	args = parser.parse_args()

	# CUDA
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('Device:', device)
	kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

	# reproducibility
	torch.manual_seed(args.seed)

	# create the model, loss, and optimizer
	model = MLP(probability = args.probability,
				device = device).to(device)
	optimizer = optim.SGD(model.parameters(),
							lr = args.lr,
							momentum = args.momentum)
	criterion = nn.CrossEntropyLoss()


	# for the CIFAR dataset
	transform = transforms.Compose([transforms.ToTensor(), 
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


	# traning
	if args.procedure == 'training':

		# data parallel
		if torch.cuda.device_count() > 1:
			print("Let's use", torch.cuda.device_count(), "GPUs!")
			model = nn.DataParallel(model)

		# dataiter = iter(train_loader)
		# images, labels = dataiter.next()
		# print(images.shape, labels.shape)

		# train and test sets
		trainset = datasets.CIFAR10(root='./data', train=True,
												download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size,
												  shuffle=False, **kwargs)

		testset = datasets.CIFAR10(root='./data', train=False,
											   download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
												 shuffle=False, **kwargs)

		for epoch in range(1, args.epochs + 1):
			train_acc = train(args, model, device, train_loader, criterion, optimizer, epoch)
			test(args, model, device, test_loader, criterion)

			# DIVNET trained till Train Acc. > 50%
			if train_acc > 50:
				break
			

		if args.save_model:
			name = 'CIFAR_' + str(args.probability) + '_batch' + str(args.train_batch_size) + '.pth'
			
			# always save as normal module
			if torch.cuda.device_count() > 1:
				torch.save(model.module.state_dict(), name)
			else:
				torch.save(model.state_dict(), name)


	# pruning and testing
	else:


		# calculate DPP by all training examples
		trainset = datasets.CIFAR10(root='./data', train=True,
												download=True, transform=transform)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000,
												  shuffle=False, **kwargs)
		train_whole_batch = enumerate(train_loader)
		assert len(list(train_loader)) == 1
		dummy_idx, (train_all_data, dummy_target) = next(train_whole_batch)
		# print(train_all_data.shape, dummy_target.shape)


		# test on all test data at once
		testset = datasets.CIFAR10(root='./data', train=False,
											   download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=10000,
												 shuffle=False, **kwargs)

		test_whole_batch = enumerate(test_loader)
		assert len(list(test_loader)) == 1
		dummy_idx, (test_all_data, target) = next(test_whole_batch)
		# print(test_all_data.shape, target.shape)

		# evaluation on the test set
		model.eval()
		test_loss = 0
		correct = 0
		reweight_test_loss = 0
		reweight_correct = 0

		# inference only
		with torch.no_grad():

			# load the model every iteration
			model.load_state_dict(torch.load(args.trained_weights, 
				map_location = torch.device('cpu')))


			# faltten all the image
			test_all_data = test_all_data.view(test_all_data.shape[0], -1)
			train_all_data = train_all_data.view(train_all_data.shape[0], -1)
			test_all_data, target = test_all_data.to(device), target.to(device)

			# get the DPP kernel and mask
			model, dpp_weight, mask = prune_MLP(model, 
												train_all_data, 
												args.pruning_choice, 
												args.reweighting, 
												args.beta, 
												args.k, 
												args.trained_weights, 
												device = device)

			# switch back to GPU
			model = model.to(device)
			output = model(test_all_data)

			# sum up batch loss
			test_loss += criterion(output, target).item()

			# get the index of the max log-probability
			pred = output.argmax(dim = 1, keepdim = True)
			correct += pred.eq(target.view_as(pred)).sum().item()


			if args.reweighting and args.pruning_choice=='dpp_edge':
	
				pruned_w1 = torch.from_numpy((mask * dpp_weight).T)
				model.w1.weight.data = pruned_w1.float().to(device)

				reweight_output = model(test_all_data)

				# sum up batch loss
				reweight_test_loss += criterion(reweight_output, target).item()

				# get the index of the max log-probability
				reweight_pred = reweight_output.argmax(dim = 1, keepdim = True)
				reweight_correct += reweight_pred.eq(target.view_as(reweight_pred)).sum().item()


		test_loss = test_loss / len(test_loader.dataset)
		print(args.pruning_choice, 'k =', args.k)
		print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
		
		if args.reweighting and args.pruning_choice=='dpp_edge':
			reweight_test_loss /= len(test_loader.dataset)

			print('\nTest set: Average reweight loss: {:.4f}, Reweighting Accuracy : {}/{} ({:.2f}%)\n'.format(
			reweight_test_loss, reweight_correct, len(test_loader.dataset),
			100. * reweight_correct / len(test_loader.dataset)))



if __name__ == '__main__':
	main()

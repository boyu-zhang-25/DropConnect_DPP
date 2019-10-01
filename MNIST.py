from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from random import sample
import math

# 2-layer MLP to compare Dropout and DropConnect
class Two_Layer_MLP(nn.Module):

	def __init__(self,
				hidden_size,
				drop_option,
				probability):
		super(Two_Layer_MLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(28 * 28, hidden_size) # 784 * hidden_size
		self.relu = nn.ReLU()
		self.drop_option = drop_option
		self.probability = probability
		self.w2 = nn.Linear(hidden_size, 10) # hidden_size * 10
		self.log_softmax = nn.LogSoftmax(dim = 1)
		self.initialize()

		# the drop layer
		if drop_option == 'out':
			print('Using Dropout with p = {}'.format(probability))
			self.dropout = nn.Dropout(p = probability)
		elif drop_option == 'connect':
			print('Using DropConnect with p = {}'.format(probability))

			# the binary sampler for mask
			self.sampler = torch.distributions.bernoulli.Bernoulli(torch.ones(self.w1.weight.shape) * probability)
		else:
			print('drop_option not supported!')
			raise ValueError

	# Xavier init
	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w2.weight.data, gain = nn.init.calculate_gain('relu'))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()

	def forward(self, x):

		# batch_size * 784 -> batch_size * hidden_size
		# switch between dropout/DropConnect
		if self.drop_option == 'out':
			x = self.relu(self.w1(x))
			# print(self.w1.weight.requires_grad)
			x = self.dropout(x)
		else:

			if self.training:
				self.drop_connect(layer_choice = 'w1')
			x = self.relu(self.w1(x))

		# batch_size * hidden_size -> batch_size * 10
		x = self.w2(x)

		# batch_size
		return self.log_softmax(x)

	# drop connect
	# TODO: no direct way of refering to weights by name in pytorch module?
	def drop_connect(self, layer_choice):
		if layer_choice == 'w1':

			# old = torch.sum(self.w1.weight.data)
			# mask = self.sampler.sample()

			# the following code is DropConnect
			'''
			mask = torch.bernoulli(self.probability * torch.ones(self.w1.weight.shape))
			# print(mask)
			# self.w1.weight.data = self.w1.weight * mask
			with torch.no_grad():
				# self.w1.weight.data.mul_(mask)
				self.w1.weight.data = self.w1.weight.data * mask
			# print(self.w1.weight.grad_fn)
			# assert old != torch.sum(self.w1.weight.data)
			'''

			# the following code is Dropout from scratch
			l = sample([i for i in range(self.w1.weight.shape[1])], math.floor((1 - self.probability) * self.w1.weight.shape[1]))
			with torch.no_grad():
				for i in l:
					self.w1.weight[:, i] = 0
					
		elif layer_choice == 'w2':
			self.w2.weight.data = self.w2.weight * self.sampler.sample()

# training loop
def train(args, model, device, train_loader, criterion, optimizer, epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):

		# faltten the image
		# torch.Size([64, 1, 28, 28]) -> [64, 28*28]
		# torch.Size([64])
		data = data.view(data.shape[0], -1)
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

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

def main():

	# hyperparameter settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')
	parser.add_argument('--log-interval', type = int, default=10, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	parser.add_argument('--drop-option', type = str,
						help='out or connect')
	parser.add_argument('--probability', type = float,
						help='probability for dropping')
	parser.add_argument('--weight-decay', type = float, default = 0.01,
						help='L2 penalty')
	parser.add_argument('--hidden-size', type = int, default = 850,
						help='hidden layer size of the two-layer MLP')

	args = parser.parse_args()
	# print(args)
	# CUDA
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

	# reproducibility
	torch.manual_seed(args.seed)

	# training data
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train = True, download = True,
					   transform = transforms.Compose([
						   transforms.ToTensor(),

						   # the mean and std of the MNIST dataset
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size = args.batch_size, shuffle=True, **kwargs)

	# testing data
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train = False, transform = transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size = args.test_batch_size, shuffle = True, **kwargs)

	# verify train/test size
	'''
	examples = enumerate(train_loader)
	batch_idx, (example_data, example_targets) = next(examples)
	print('# of training batches: {}; data shape: {}; target shape: {}'.format(len(list(examples)), example_data.shape, example_targets.shape))
	examples = enumerate(test_loader)
	batch_idx, (example_data, example_targets) = next(examples)
	print('# of testing batches: {}; data shape: {}; target shape: {}'.format(len(list(examples)), example_data.shape, example_targets.shape))
	'''

	# create the model, loss, and optimizer
	model = Two_Layer_MLP(hidden_size = args.hidden_size,
							drop_option = args.drop_option,
							probability = args.probability).to(device)
	optimizer = optim.SGD(model.parameters(),
							lr = args.lr,
							weight_decay = args.weight_decay,
							momentum = args.momentum)

	# optimizer = optim.Adam(model.parameters(), weight_decay = 0.01)
	criterion = nn.NLLLoss(reduction = 'sum')

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, criterion, optimizer, epoch)
		test(args, model, device, test_loader, criterion)

	if (args.save_model):
		if args.drop_option == 'out':
			torch.save(model.state_dict(),"mnist_two_layer_dropout.pt")
		else:
			torch.save(model.state_dict(),"mnist_two_layer_DropConnect.pt")

if __name__ == '__main__':
	main()

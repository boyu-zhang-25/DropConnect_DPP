from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from random import sample
import math

# 2-layer MLP to compare Dropout and DropConnect
# implicit regularization applied during training
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
		self.w2 = nn.Linear(hidden_size, 10) # 10 * hidden_size
		self.device = device
		self.initialize()

		# the drop layer
		if drop_option == 'out':
			print('Using Dropout with p = {}'.format(probability))
			self.dropout = nn.Dropout(p = probability)
		elif drop_option == 'connect':
			print('Using DropConnect with p = {}'.format(probability))
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
			x = self.dropout(x)
		else:

			# only apply during training
			if self.training:
				x = self.drop_connect(x, layer_choice = 'w1')
				x = self.relu(x)
			else:
				x = self.relu(self.w1(x))

		# batch_size * hidden_size -> batch_size * 10
		x = self.w2(x)
		# print(self.w1.weight.grad_fn, self.w2.weight.grad_fn, x.requires_grad)
		# batch_size * 10
		return x

	# drop connect
	# different masks for each example in the same batch
	def drop_connect(self, x, layer_choice):

		# [batch_size, hidden_size]
		result = torch.zeros(x.shape[0], self.hidden_size).to(self.device)

		'''
		# fully vectorized
		# binary masks [batch, hidden_size, 784]
		mask = torch.bernoulli(self.probability * torch.ones(
															x.shape[0],
															self.w1.weight.shape[0],
															self.w1.weight.shape[1]))
		if layer_choice == 'w1':

			# expand the weight
			# [hidden_size, 784] ->
			self.w1 = self.w1.weight.unsqueeze(2).repeat()
		'''

		# [hidden_size, 784, batch_size]
		mask = torch.bernoulli(self.probability * torch.ones(
															self.w1.weight.shape[0],
															self.w1.weight.shape[1],
															x.shape[0])).to(self.device)
		if layer_choice == 'w1':


			# TODO: vectorization
			# for each example in the batch
			for batch in range(x.shape[0]):

				old_weight = self.w1.weight
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
	parser.add_argument('--hidden-size', type = int, default = 300,
						help='hidden layer size of the two-layer MLP')

	args = parser.parse_args()
	# print(args)
	# CUDA
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)
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
	model = MLP(hidden_size = args.hidden_size,
							drop_option = args.drop_option,
							probability = args.probability,
							device = device).to(device)
	optimizer = optim.SGD(model.parameters(),
							lr = args.lr,
							momentum = args.momentum)
	criterion = nn.CrossEntropyLoss()

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, criterion, optimizer, epoch)
		test(args, model, device, test_loader, criterion)

	if (args.save_model):
		if args.drop_option == 'out':
			torch.save(model.state_dict(),"mnist_two_layer_Dropout.pt")
		else:
			torch.save(model.state_dict(),"mnist_two_layer_DropConnect.pt")

if __name__ == '__main__':
	main()

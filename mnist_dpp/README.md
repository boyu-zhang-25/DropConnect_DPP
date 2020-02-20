# Understanding Diversity based Edge and Node Pruning of Neural Networks
Code for the paper *Understanding Diversity based Edge and Node Pruning of Neural Networks*

# Requirements
```
torch==1.3.1
torchvision==0.4.2
dppy==0.2.0
numpy==1.17.2
jupyter==1.0.0
scikit-learn==0.20.2
```
It is suggested to create a python virtual env with the above dependencies. It should be very easy.


## To perform DPP Edge and DPP Node on two-layer MNIST MLP

To train without Dropout
>python3 MNIST.py --procedure training --epochs 40 --lr .07

To train with Dropout
>python3 MNIST.py --procedure training --epochs 45 --lr .2 --drop-option 'out' --probability .5

To prune the network
>python3 MNIST.py --procedure test --pruning_choice dpp_node --k 250 --reweighting --trained_weights mnist_two_layer_Dropout.pt


with the following arguments
```	
	parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
						help='input batch size for training (default: 1000)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0.9, metavar='M',
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
						help='out for dropout; otherwise do not use this flag')
	parser.add_argument('--probability', type = float, default = 0.5, 
						help='probability for dropout')
	parser.add_argument('--hidden-size', type = int, default = 500,
						help='hidden layer size of the two-layer MLP')
	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, or dpp_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 20,
						help='number of edges/nodes to preserve')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training or purning')
	parser.add_argument('--trained_weights', type = str, default = 'mnist_two_layer.pt',
						help='path to the trained weights for loading')
	parser.add_argument('--reweighting', action='store_true', default = False,
						help='For fusing the lost information')
```

Node-Edge correspondence (`--k`):

Given hidden size as 500 and the input dimension as 784 (flatten image), the number of remaining weghts are

|Node   |Edge   |
|---	|---	|---
|50   	|69   	|10%   	|
|100 	|148 	|20%   	|
|150  	|228  	|30%   	|
|200	|307   	|40%   	|
|250	|387   	|50%   	|
|300	|466   	|60%   	|
|350	|545   	|70%   	|
|400	|625   	|80%   	|



# DropConnect_DPP
Diversifying Neural Network Connections via Determinantal Point Processes

# Requirements
```
python3
torch==1.2.0
torchvision==0.4.0
```

## To test DPP purning in teacher-student setup

Generating dataset and the teacher network:
>python3 teacher_dataset.py --input_dim 1000 --teacher_h_size 2 --teacher_path teacher.pkl --num_data 80000 --mode normal  --sig_w 0

with the following arguments:
```
	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--teacher_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--num_data', type = int, help='Number of data points to be genrated.')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')
	parser.add_argument('--sig_w', type = float, help='scaling variable for the output noise.')

	# data storage
	parser.add_argument('--teacher_path', type = str, help='Path to store the teacher network (dataset).')
```

Training the student network:
>python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid  --mode normal  --epoch 1 --lr 0.05

Pruning the student network:
>python3 teacher_student.py --input_dim 500 --student_h_size 6  --teacher_path teacher.pkl  --nonlinearity sigmoid --pruning_choice dpp_node  --mode normal  --trained_weights student_6.pth --procedure purning --num_masks 100

with the following arguments:
```
	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--nonlinearity', type = str, help='choice of the activation function')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')

	# optimization setup
	parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# pruning parameters
	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, dpp_node, random_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 5,
						help='number of edges/nodes to preserve')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training or purning')
	parser.add_argument('--reweighting', action='store_true', default = False,
						help='For fusing the lost information')
	parser.add_argument('--num_masks', type = int, default = 1,
						help='Number of masks to be sampled.')

	# data storage
	parser.add_argument('--trained_weights', type = str, default = 'place_holder', help='path to the trained weights to be loaded')
	parser.add_argument('--teacher_path', type = str, help='Path to the teacher network (dataset).')
```

## To test random Dropout and random DropConnect on the two-layer MNIST MLP
>python3 MNIST.py

with the following arguments:
```
batch_size=1000
drop_option='connect'
epochs=10
hidden_size=850
log_interval=10
lr=0.0001
momentum=0
no_cuda=False
probability=0.1
save_model=False
seed=1
test_batch_size=1000
weight_decay=0.01
procedure='pruning' or 'training': post pruning or pure training
pruning_choice='dpp_edge' or 'dpp_node'
trained_weights='mnist_two_layer.pt': path to the trained, to-be-pruned model
```

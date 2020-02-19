# DropConnect_DPP
Diversifying Neural Network Connections via Determinantal Point Processes

# Requirements
```
python3
torch==1.2.0
torchvision==0.4.0
```

## To perform DPP purning and simulations in the teacher-student setup

Generating dataset and the teacher network:
>python3 teacher_dataset.py --input_dim 500 --teacher_h_size 2 --teacher_path teacher.pkl --num_data 800000 --mode normal --sig_w 0 --v_star 4

with the following arguments:
```
	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--teacher_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--num_data', type = int, help='Number of data points to be genrated.')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')
	parser.add_argument('--sig_w', type = float, help='scaling variable for the output noise.')
	parser.add_argument('--v_star', type = int, help='ground truth second layer weight')

	# data storage
	parser.add_argument('--teacher_path', type = str, help='Path to store the teacher network (dataset).')
```

Training the student network:
>python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid  --mode normal  --epoch 1 --lr 0.5

Pruning the student network:
>python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid --pruning_choice dpp_node  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 100 --k 3


with the following arguments:
```
	# network parameter
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--nonlinearity', type = str, help='choice of the activation function')
	parser.add_argument('--mode', type = str, help='soft_committee or normal')

	# optimization setup
	parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
						help='learning rate (default: 0.0001)')
	parser.add_argument('--momentum', type=float, default = 0, metavar='M',
						help='SGD momentum (default: 0)')
	parser.add_argument('--epoch', type = int, default = 1, help='number of epochs')
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)')

	# pruning parameters
	parser.add_argument('--pruning_choice', type = str, default = 'dpp_edge',
						help='pruning option: dpp_edge, random_edge, dpp_node')
	parser.add_argument('--beta', type = float, default = 0.3,
						help='beta for dpp')
	parser.add_argument('--k', type = int, default = 2,
						help='number of parameters to preserve (for dpp_node: # of nodes; for dpp_edge: # of weights per node)')
	parser.add_argument('--procedure', type = str, default = 'training',
						help='training or pruning')
	parser.add_argument('--reweighting', action='store_true', default = False,
						help='For fusing the lost information')
	parser.add_argument('--num_masks', type = int, default = 1,
						help='Number of masks to be sampled.')
	# data storage
	parser.add_argument('--trained_weights', type = str, default = 'place_holder', help='path to the trained weights for loading')
	parser.add_argument('--teacher_path', type = str, help='Path to the teacher network (dataset).')
```

For `dpp_edge`, it automatically saves the kernels created for each node in a list into a pickle file. Switch the flag `load_from_pkl` in the `dpp_sample_edge_ts` method to save time if you want to run again.


To get the order parameters (Q, T, R) of the networks:
>python3 evaluate.py --path_to_student_mask student_masks_dpp_node_6_3.pkl --path_to_teacher teacher.pkl --input_dim 500

where the output pickle from the previous pruning process is named as `'student_masks_' + args.pruning_choice + '_' + str(args.student_h_size) + "_" + str(args.k) + '.pkl'`.


## To compare dpp_node and dpp_edge on the test dataset

>python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid --pruning_choice dpp_edge  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 100 --k 50

stricly followed by

>python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid  --mode normal  --trained_weights student_6.pth --procedure testing --pruning_choice dpp_edge --k 50


Change the argument `--pruning_choice` to compare between `dpp_node` and `dpp_edge`.
NOTICE: Run the above command consecutively; keep the `--student_h_size`, `--k`, and `--pruning_choice` consistent and correct; be CAREFUL with `--procedure` and the `--input_dim` when calculating the number of parameters.

Node-Edge correspondence (`--k`):

Given 6 student nodes and the input dimension, the number of remaining weghts are

|Node   |Edge (inp_dim = 100)  	|Edge (inp_dim = 500)  	|
|---	|---	|---
|1   	|16   	|83   	|
|2  	|33 	|166   	|
|3  	|50  	|250   	|
|4  	|66   	|333   	|
|5  	|83   	|417   	|


## To test DPP Edge and DPP Node on two-layer MNIST MLP


## Fixing the finite DPP sampling of the dppy package
In `site-packages/dppy/exact_sampling.py", line 531, in proj_dpp_sampler_eig_GS`, numpy may give a normalization problem, which is still unsolved. There is a simple solution for our case.

Modification:
```
    for it in range(size):
        # Pick an item \propto this squred distance
        arr = np.abs(norms_2[avail]) / (rank - it)
        arr = arr / np.sum(arr)
        j = rng.choice(ground_set[avail], p=arr)
```

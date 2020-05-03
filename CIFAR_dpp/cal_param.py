import math

def cal_param_CIFAR(input_dim, hid_dim):

	percent = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	node = [int(p * hid_dim) for p in percent]

	edge = [math.ceil(n * input_dim / hid_dim) for n in node]
	print(node)
	print(edge)


cal_param_CIFAR(3*32*32, 1000)
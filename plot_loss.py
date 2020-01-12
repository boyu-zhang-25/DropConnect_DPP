import numpy as np 
import matplotlib.pyplot as plt

# SH = 6
# please run the teacher_student.py as instructed by README to reproduce

def calculate_param(student_hidden_size, inp_dim):

	nodes = [k + 1 for k in range(student_hidden_size - 1)]
	edge = [(k * inp_dim + k - student_hidden_size) / student_hidden_size for k in nodes]

	total_edge = student_hidden_size * inp_dim + student_hidden_size
	random = [(inp_dim * k + k) / total_edge for k in nodes]

	print(nodes)
	print(edge)
	print(random)

# dpp_node
node_loss = [9.596591002260262, 9.484469927823054, 9.431387147356173, 9.257694226096528, 8.857213005378735]

# dpp edge
edge_loss = [9.484352296676201, 9.434764798012361, 9.387114756486044, 9.005537451073242, 8.661559665117923]

# xticks = ['1:16', '2:33', '3:50', '4:66', '5:83']

def plot_loss(xticks):

	plt.plot(node_loss, label = "Avg. DPP_Node Loss")
	plt.plot(edge_loss, label = "Avg. DPP_Edge Loss")
	plt.legend(loc = "best")

	x = [k for k in range(5)]
	plt.xticks(x, xticks)
	plt.xlabel('Parameters remained (Node V.S. Edge)')
	plt.ylabel('Average Loss')

	# plt.tight_layout()
	plt.grid(True)
	plt.title('Average Test Loss')
	plt.savefig('Average Test Loss', dpi = 150)

# plot_loss(xticks)
calculate_param(student_hidden_size = 6, inp_dim = 500)




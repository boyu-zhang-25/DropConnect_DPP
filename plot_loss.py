import numpy as np 
import matplotlib.pyplot as plt
from loss_stat import *
plt.rcParams.update({'font.size': 14})

# SH = 6
# please run the teacher_student.py as instructed by README to reproduce


xticks = ['1:83', '2:166', '3:250', '4:333', '5:417']

# edge
all_edge_noiseless = np.asarray(all_edge_noiseless)
noiseless_edge_std = np.std(all_edge_noiseless, axis=0)
noiseless_edge_means = np.mean(all_edge_noiseless, axis=0)	

all_edge_noise0_25 = np.asarray(all_edge_noise0_25)
noise0_25_edge_std = np.std(all_edge_noise0_25, axis=0)
noise0_25_edge_means = np.mean(all_edge_noise0_25, axis=0)

# node
all_node_noiseless = np.asarray(all_node_noiseless)
noiseless_node_std = np.std(all_node_noiseless, axis=0)
noiseless_node_means = np.mean(all_node_noiseless, axis=0)	

all_node_noise0_25 = np.asarray(all_node_noise0_25)
noise0_25_node_std = np.std(all_node_noise0_25, axis=0)
noise0_25_node_means = np.mean(all_node_noise0_25, axis=0)

# rand edge
all_rand_noiseless = np.asarray(all_rand_noiseless)
noiseless_rand_std = np.std(all_rand_noiseless, axis=0)
noiseless_rand_means = np.mean(all_rand_noiseless, axis=0)	

all_rand_noise0_25 = np.asarray(all_rand_noise0_25)
noise0_25_rand_std = np.std(all_rand_noise0_25, axis=0)
noise0_25_rand_means = np.mean(all_rand_noise0_25, axis=0)

x = [k for k in range(5)]

plt.figure(1)
plt.errorbar(x, noiseless_edge_means, yerr=noiseless_edge_std, marker = '^', label="Avg. DPP_Edge Loss")
plt.errorbar(x, noiseless_node_means, yerr=noiseless_node_std, marker = 's', label="Avg. DPP_Node Loss")
plt.errorbar(x, noiseless_rand_means, yerr=noiseless_rand_std, marker = 'o', label="Avg. Rand_Edge Loss")

print("noiseless edge data")
print(noiseless_edge_means)
print(noiseless_edge_std)

print("noiseless node data")
print(noiseless_node_means)
print(noiseless_node_std)

print("noiseless rand edge data")
print(noiseless_rand_means)
print(noiseless_rand_std)

plt.plot(x, [noiseless_unpruned_loss for _ in range(5)], dashes=[6, 2], label="Unpruned Loss")
plt.legend(loc = "best")

plt.xticks(x, xticks)
plt.xlabel('Parameters remained (Node V.S. Edge)')
plt.ylabel('Average Loss')
# plt.tight_layout()
# plt.grid(True)
plt.savefig('Noiseless Average Test Loss', dpi = 200)

plt.figure(2)
plt.errorbar(x, noise0_25_edge_means, yerr=noise0_25_edge_std, marker = '^', label="Avg. DPP_Edge Loss")
plt.errorbar(x, noise0_25_node_means, yerr=noise0_25_node_std, marker = 's', label="Avg. DPP_Node Loss")
plt.errorbar(x, noise0_25_rand_means, yerr=noise0_25_rand_std, marker = 'o', label="Avg. Rand_Edge Loss")

print("noisy edge data")
print(noise0_25_edge_means)
print(noise0_25_edge_std)

print("noisy node data")
print(noise0_25_node_means)
print(noise0_25_node_std)

print("noisy rand edge data")
print(noise0_25_rand_means)
print(noise0_25_rand_std)

plt.plot(x, [noise0_25_unpruned_loss for _ in range(5)], dashes=[6, 2], label="Unpruned Loss")
plt.legend(loc = "best")

plt.xticks(x, xticks)
plt.xlabel('Parameters remained (Node V.S. Edge)')
plt.ylabel('Average Loss')
# plt.tight_layout()
# plt.grid(True)
plt.savefig('Noise 0_25 Average Test Loss', dpi = 200)





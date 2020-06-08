from evaluate import *



def valid_choice(student_h_size,input_dim,pruning_type):

	if pruning_type == 'node':
		return [i+1 for i in range(student_h_size)]
	elif pruning_type == 'edge':
		return [int((i+1)*input_dim/student_h_size) for i in range(student_h_size)]



def main():

	parser = argparse.ArgumentParser(description='Order Parameter')
	parser.add_argument('--pruning_choice', type = str)
	parser.add_argument('--path_to_teacher', type = str, default = 'place_holder')
	parser.add_argument('--input_dim', type = int, help='The input dimension for each data point.')
	parser.add_argument('--student_h_size', type = int, help='hidden layer size of the student MLP')
	parser.add_argument('--teacher_h_size', type = int, help='hidden layer size of the teacher MLP')
	args = parser.parse_args()

	student_path = 'student_masks_'+args.pruning_choice+'_'+str(args.student_h_size)+'_'

	#pruned_sizes = valid_choice(args.student_h_size,args.input_dim,args.pruning_choice[-4:]) 
	pruned_sizes = [83, 166, 250, 333, 417, 500]
	width = (args.student_h_size+args.teacher_h_size)*args.student_h_size
	height = args.student_h_size

	width_ratios = list(np.array([[args.student_h_size,args.teacher_h_size] for sub in range(args.student_h_size)]).flatten())
	#width_ratios.append(0.5)
	
	fig,ax = plt.subplots(ncols=2*args.student_h_size,figsize=(width, height),gridspec_kw={'width_ratios': width_ratios},sharey=True)


	for prune_ind,num_prune in enumerate(pruned_sizes):
		cur_student_path = student_path + str(num_prune) +'.pkl'

		expected_Q, unpruned_Q, teacher_Q = get_Q(cur_student_path, args.path_to_teacher, args.input_dim)	
		expected_R, unpruned_R = get_R(cur_student_path, args.path_to_teacher, args.input_dim)
		# estimated_Q, unpruned_Q = get_cube_Q(args.path_to_student_mask, args.path_to_teacher, args.input_dim)

	

		# Permute the matrix to make it block diagonal
		student_hid_dim, teacher_hid_dim = unpruned_R.shape
		z = int(student_hid_dim/teacher_hid_dim)
		unpruned_R_dash, unpruned_Q_dash, expected_R_dash ,expected_Q_dash = np.zeros((student_hid_dim,teacher_hid_dim)),  np.zeros((student_hid_dim,student_hid_dim)), np.zeros((student_hid_dim,teacher_hid_dim)),  np.zeros((student_hid_dim,student_hid_dim))
		dic = [[] for x in range(teacher_hid_dim)]
		for i in range(teacher_hid_dim):
			for j in range(student_hid_dim):
				if abs(unpruned_R[j][i])>=0.7:
					dic[i].append(j)

		print(dic,"hello")
		for x in range(teacher_hid_dim):
			for y in range(len(dic[x])):
				new_row = x*z+y
				cur = dic[x][y]
				print(new_row,cur)
				unpruned_R_dash[new_row,:] = unpruned_R[cur,:]
				expected_R_dash[new_row,:] = expected_R[cur,:]

		for x in range(student_hid_dim):
			for y in range(x+1):

				i = dic[int(x/z)][x%z]
				j = dic[int(y/z)][y%z]

				if x==y:
					unpruned_Q_dash[x][x] =  unpruned_Q[i][i]
					expected_Q_dash[x][x] =  expected_Q[i][i]
				else:
					unpruned_Q_dash[x][y] = unpruned_Q[i][j]
					unpruned_Q_dash[y][x] = unpruned_Q[i][j]

					expected_Q_dash[x][y] = expected_Q[i][j]
					expected_Q_dash[y][x] = expected_Q[i][j]

		
		im = ax[2*prune_ind].imshow(abs(expected_Q_dash), vmin=0, vmax=1)
		im1 = ax[2*prune_ind+1].imshow(abs(expected_R_dash), vmin=0, vmax=1)

		# im = ax[2*prune_ind].imshow(expected_Q, vmin=0, vmax=1)
		# im1 = ax[2*prune_ind+1].imshow(expected_R, vmin=0, vmax=1)
	
	# plot_Q(expected_Q_dash, unpruned_Q_dash, teacher_Q)
	# plot_R(expected_R_dash, unpruned_R_dash)
	#fig.colorbar(im, cax=ax[-1])
	plt.savefig('all_'+args.pruning_choice+'.pdf')

if __name__ == '__main__':
	main()





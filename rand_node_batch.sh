filename="rand_node_output.txt"
vars='1 2 3 4 5'
rounds='1 2 3'
teacher_path='teacher_noise0.25.pkl'
for round in $rounds
do
	filename="rand_node_output_lr0.5_v4_800000_noise0.25_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid --pruning_choice random_node  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 20 --k $var >> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid  --mode normal  --trained_weights student_6.pth --procedure testing --pruning_choice random_node --k $var >> $filename
	done
done
filename="rand_edge_output.txt"
vars='24 49 74 99 124'
rounds='1 2 3 4 5'
teacher_path='teacher_5.pkl'
for round in $rounds
do
	filename="rand_edge_output_lr0.5_v4_1500000_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 20 --teacher_path $teacher_path  --nonlinearity sigmoid --pruning_choice random_edge  --mode normal  --trained_weights student_20.pth --procedure pruning --num_masks 25 --k $var >> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 20 --teacher_path $teacher_path  --nonlinearity sigmoid  --mode normal  --trained_weights student_20.pth --procedure testing --pruning_choice random_edge --k $var >> $filename
	done
done
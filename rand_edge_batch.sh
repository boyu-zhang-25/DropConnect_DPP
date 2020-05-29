filename="rand_edge_rwt_output.txt"
vars='83'
rounds='1'
teacher_path='teacher.pkl'
for round in $rounds
do
	filename="rand_edge_rwt_output_lr0.5_v4_800000_noiseless_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid --pruning_choice random_edge  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 50 --k $var >> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid  --mode normal  --trained_weights student_6.pth --procedure testing --pruning_choice random_edge --k $var >> $filename
	done
done
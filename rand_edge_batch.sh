filename="rand_edge_output.txt"
vars='83 166 250 333 417'
rounds='1 2 3'
teacher_path='teacher.pkl'
rwt='1'
for round in $rounds
do
	filename="rand_edge_rwt_output_lr0.5_v4_800000_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid --pruning_choice random_edge  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 10 --k $var >> $filename
		python3 teacher_student.py --input_dim 500 --student_h_size 6 --teacher_path $teacher_path  --nonlinearity sigmoid  --mode normal  --trained_weights student_6.pth --procedure testing --pruning_choice random_edge --k $var --reweighting $rwt >> $filename
	done
done
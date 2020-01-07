filename="dpp_node_output.txt"
vars='1 2 3 4 5'
for var in $vars
do
    echo '============================================================' 2>> $filename
    python3 teacher_student.py --input_dim 100 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid --pruning_choice dpp_node  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 100 --k $var >> $filename
    python3 teacher_student.py --input_dim 100 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid  --mode normal  --trained_weights student_6.pth --procedure testing --pruning_choice dpp_node --k $var >> $filename
done

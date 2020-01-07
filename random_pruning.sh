filename="random_output.txt"
vars='1 2 3 4 5'
for var in $vars
do
    echo '============================================================' 2>> $filename
    python3 teacher_student.py --input_dim 100 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid --pruning_choice random_edge  --mode normal  --trained_weights student_6.pth --procedure pruning --num_masks 1 --k $var >> $filename
    python3 teacher_student.py --input_dim 100 --student_h_size 6 --teacher_path teacher.pkl  --nonlinearity sigmoid  --mode normal  --trained_weights student_6.pth --procedure testing --pruning_choice random_edge --k $var >> $filename
done

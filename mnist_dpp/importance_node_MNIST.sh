vars='50 100 150 200 250 300 350 400 450'
rounds='1'
for round in $rounds
do
	filename="importance_node_MNIST_0.0_batch1000_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 MNIST.py --pruning_choice importance_node --k $var --procedure pruning >> $filename
	done
done
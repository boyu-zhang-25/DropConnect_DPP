vars='50 100 150 200 250 300 350 400 450'
rounds='1 2 3 4 5 6 7 8 9 10'
for round in $rounds
do
	filename="dpp_node_MNIST_0.0_batch_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 MNIST.py --pruning_choice dpp_node --k $var --procedure pruning >> $filename
	done
done
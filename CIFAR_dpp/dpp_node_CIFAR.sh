vars='100 200 300 400 500 600 700 800 900'
rounds='1 2 3 4 5 6 7 8 9 10'
for round in $rounds
do
	filename="dpp_node_CIFAR_0.0_batch128_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 CIFAR.py --pruning_choice dpp_node --k $var --procedure pruning >> $filename
	done
done
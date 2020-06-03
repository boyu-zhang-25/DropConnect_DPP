vars='320 396 471 547 622 698 773 848 924'
rounds='1 2 3 4 5'
for round in $rounds
do
	filename="rand_node_CIFAR_0.0_batch128_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 CIFAR.py --pruning_choice random_node --k $var --procedure pruning >> $filename
	done
done
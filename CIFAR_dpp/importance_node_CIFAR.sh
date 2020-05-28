vars='320 396 471 547 622 698 773 848 924'
rounds='1'
for round in $rounds
do
	filename="importance_node_rwt_CIFAR_0.0_batch128_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 CIFAR.py --pruning_choice importance_node --k $var --procedure pruning >> $filename
	done
done
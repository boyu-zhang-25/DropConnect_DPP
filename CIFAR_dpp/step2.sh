vars='320 396 471 547 622 698 773 848 924'
rounds='1 2 3'
rwt='1'
for round in $rounds
do
	filename="dpp_node_rwt_CIFAR_0.0_batch128_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 CIFAR.py --pruning_choice dpp_node --k $var --procedure pruning --reweighting $rwt >> $filename
	done
done
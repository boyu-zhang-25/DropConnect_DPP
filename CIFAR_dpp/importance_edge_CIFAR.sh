vars='308 615 922 1229 1536 1844 2151 2458 2765'
rounds='1'
for round in $rounds
do
	filename="importance_edge_CIFAR_0.0_batch128_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 CIFAR.py --pruning_choice importance_edge --k $var --procedure pruning >> $filename
	done
done
# vars='320 396 471 547 622 698 773 848 924'
# rounds='6 7 8 9 10'
rwt='0'
# for round in $rounds
# do
# 	filename="dpp_node_CIFAR_0.0_batch128_output_round${round}.txt"
# 	for var in $vars
# 	do
# 		echo '============================================================' 2>> $filename
# 		python3 CIFAR.py --pruning_choice dpp_node --k $var --procedure pruning --reweighting $rwt >> $filename
# 	done
# done
vars='308 615 922 1229 1537 1844 2151 2458 2765'
rounds='1 2 3 4 5'
for round in $rounds
do
	filename="dpp_edge_CIFAR_0.0_batch128_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 CIFAR.py --pruning_choice dpp_edge --k $var --procedure pruning --reweighting $rwt >> $filename
	done
done
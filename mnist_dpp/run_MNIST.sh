vars='79 157 236 314 392 471 549 628 706'
rounds='1 2 3'
rwt='1'
for round in $rounds
do
	filename="dpp_edge_MNIST_0.0_batch1000_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 MNIST.py --pruning_choice dpp_edge --k $var --procedure pruning >> $filename
	done
done
for round in $rounds
do
	filename="dpp_edge_MNIST_0.0_batch1000_rwt_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 MNIST.py --pruning_choice dpp_edge --k $var --procedure pruning --reweighting $rwt >> $filename
	done
done
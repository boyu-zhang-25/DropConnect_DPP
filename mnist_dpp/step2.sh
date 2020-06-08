# vars='225 255 286 316 347 377 408 438 469'
# rounds='11 12 13'
# for round in $rounds
# do
# 	filename="dpp_node_MNIST_0.0_batch1000_output_round${round}.txt"
# 	for var in $vars
# 	do
# 		echo '============================================================' 2>> $filename
# 		python3 MNIST.py --pruning_choice dpp_node --k $var --procedure pruning >> $filename
# 	done
# done
vars='79 157 236 314 392 471 549 628 706'
rounds='1 2 3 4 5'
beta='0.5'
for round in $rounds
do
	filename="dpp_edge_MNIST_0.0_batch1000_output_round${round}.txt"
	for var in $vars
	do
		echo '============================================================' 2>> $filename
		python3 MNIST.py --pruning_choice dpp_edge --k $var --procedure pruning --beta $beta>> $filename
	done
done
# for round in $rounds
# do
# 	filename="rand_edge_MNIST_0.0_batch1000_output_round${round}.txt"
# 	for var in $vars
# 	do
# 		echo '============================================================' 2>> $filename
# 		python3 MNIST.py --pruning_choice random_edge --k $var --procedure pruning >> $filename
# 	done
# done
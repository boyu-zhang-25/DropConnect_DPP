edge_var='79'
node_var='225'
python3 MNIST.py --pruning_choice dpp_node --k $node_var --procedure pruning
python3 MNIST.py --pruning_choice dpp_edge --k $edge_var --procedure pruning
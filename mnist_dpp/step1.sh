edge_var='706'
node_var='469'
beta='0.1'
python3 MNIST.py --pruning_choice dpp_node --k $node_var --procedure pruning --beta $beta
beta='0.1'
python3 MNIST.py --pruning_choice dpp_edge --k $edge_var --procedure pruning --beta $beta
#!/bin/bash

source /home/anumoshsad/anaconda3/bin/activate dpp_drop

filename="./Results/5runsDppNode_mnist_without_dropout.txt"
vars='22 54 38 69 101 132 164 195 227 258 290 321'
echo '============================================================' >> $filename
echo 'with dpp_node ' >> $filename
echo '============================================================' >> $filename


for var in $vars
do
    for i in 1 2 3 4 5
    do
        echo k= $var >> $filename
        python MNIST.py --procedure test --pruning_choice dpp_node --k $var >> $filename
    done
done
echo '============================================================' >> $filename
echo 'with dpp_node on dropout model' >> $filename
echo '============================================================' >> $filename

for var in $vars
do
    for i in 1 2 3 4 5
    do
        echo k= $var >> $filename
        python MNIST.py --procedure test --pruning_choice dpp_node --k $var --trained_weights mnist_two_layer_Dropout.pt >> $filename
    done
done

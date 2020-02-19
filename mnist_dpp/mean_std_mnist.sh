#!/bin/bash

source /home/anumoshsad/anaconda3/bin/activate dpp_drop

filename="./Results/50percent_MNIST_fixed.txt"
#vars='69 148 228 307 387 466 545 625 704 784'
#vars='704 784'
# vars='30 69 109 148'
vars='387'
echo '============================================================' >> $filename
echo 'with dpp_edge' >> $filename
echo '============================================================' >> $filename


for var in $vars
do
    for i in 1 2
    do
        echo k= $var >> $filename
        python MNIST.py --procedure test --pruning_choice dpp_edge --k $var --reweighting  >> $filename
    done
done
vars='228 307'
echo '============================================================' >> $filename
echo 'with dpp_edge on dropout model' >> $filename
echo '============================================================' >> $filename

for var in $vars
do
    for i in 1 2
    do
        echo k= $var >> $filename
        python MNIST.py --procedure test --pruning_choice dpp_edge --k $var --reweighting --trained_weights mnist_two_layer_Dropout.pt >> $filename
    done
done

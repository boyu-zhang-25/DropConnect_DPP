#!/bin/bash

source /home/anumoshsad/anaconda3/bin/activate dpp_drop

filename="./Results/node_without_dropout_hiK_mnist_fixed_rewt.txt"
#vars='25 50 75 100 150 200 250 300 350 400 450 500'
vars='150 200 250 300 350 400'
echo '============================================================' >> $filename
echo 'with dpp_node ' >> $filename
echo '============================================================' >> $filename
echo '============================================================' >> $filename
echo 'without reweighting' >> $filename
echo '============================================================' >> $filename
for var in $vars
do
    for i in 1 2 3 4
    do
        echo k= $var >> $filename
       python MNIST.py --procedure test --pruning_choice dpp_node --k $var >> $filename
    done
done
echo '============================================================' >> $filename
echo 'with dpp_node on dropout model without reweighting' >> $filename
echo '============================================================' >> $filename

for var in $vars
do
   for i in 1 2 3 4
   do
       echo k= $var >> $filename
       python MNIST.py --procedure test --pruning_choice dpp_node --k $var --trained_weights mnist_two_layer_Dropout.pt >> $filename
   done
done
echo '============================================================' >> $filename
echo 'node without dropout with reweighting' >> $filename
echo '============================================================' >> $filename
for var in $vars
do
   for i in 1 2 3 4
   do
       echo k= $var >> $filename
       python MNIST.py --procedure test --pruning_choice dpp_node --k $var --reweighting  >> $filename
   done
done
echo '============================================================' >> $filename
echo 'with reweighting on dropout model' >> $filename
echo '============================================================' >> $filename

for var in $vars
do
   for i in 1 2 3 4
   do
       echo k= $var >> $filename
       python MNIST.py --procedure test --pruning_choice dpp_node --k $var --reweighting --trained_weights mnist_two_layer_Dropout.pt >> $filename
   done
done

#!/bin/bash

source /home/anumoshsad/anaconda3/bin/activate dpp_drop

filename="output_b.txt"
vars='250 300 350 400 450'
for var in $vars
do
    echo k= $var >> $filename
    python MNIST.py --procedure test --pruning_choice dpp_edge --k $var >> $filename
done
echo '============================================================' 2>> $filename
echo 'with reweighting' 2>> $filename
echo '============================================================' 2>> $filename
for var in $vars
do
    echo k= $var 2>> $filename
    python MNIST.py --procedure test --pruning_choice dpp_edge --k $var --reweighting 2>> $filename
done
echo All done
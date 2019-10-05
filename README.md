# DropConnect_DPP
Diversifying Neural Network Connections via Determinantal Point Processes

# Requirements
```
python3
torch==1.2.0
torchvision==0.4.0
```
## To test random Dropout and random DropConnect on the two-layer MLP
>python3 MNIST.py

with the following arguments:
```
batch_size=1000
drop_option='connect'
epochs=10
hidden_size=850
log_interval=10
lr=0.0001
momentum=0
no_cuda=False
probability=0.1
save_model=False
seed=1
test_batch_size=1000
weight_decay=0.01
procedure='pruning' or 'training': post pruning or pure training
pruning_choice='dpp_edge' or 'dpp_node'
trained_weights='mnist_two_layer.pt': path to the trained, to-be-pruned model
```

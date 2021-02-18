Results on MNIST and CIFAR-10 for UAI

Each pickle contains a list of __Test Errors__ in the __order__ of:

```
dpp_edge_rwt_mean,
dpp_edge_rwt_std, 
dpp_node_rwt_mean, 
dpp_node_rwt_std, 
rand_edge_rwt_mean, 
rand_edge_rwt_std,
rand_node_rwt_mean, 
rand_node_rwt_std,
imp_node_rwt, 
imp_edge_rwt, 
original_err # repeated 9 times as a numpy array
```

The order is important. All the average test errors (and standard deviations) are in numpy array of length 9 (20% to 100% parameter left after pruning). 
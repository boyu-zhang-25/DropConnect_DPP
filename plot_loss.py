# SH = 6
unpruned_loss = 0.06630606506217464

# dpp_node
11.477771255565537 # k = 1
11.429267209574437 # k = 2
11.366792595131999 # k = 3
10.921788625507526 # k = 4
10.743978358242781 # k = 5
0.06630606506217464 # k = 6

# dpp edge
'''
edge = (node * inp_dim + node - student_hidden_size) / student_hidden_size
when inp_dim = 100, student_hidden_size = 6
for node: 1 to 5, edges per node are:
15.833333333333334
32.666666666666664
49.5
66.33333333333333
83.16666666666667
'''


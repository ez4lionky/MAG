import numpy as np
with open('acc_results.txt') as f:
    sum = []
    line = f.readline()
    while line:
        sum.append(line)
        print(line)
        line = f.readline()
    sum = np.array(sum, dtype='float64')
    print('The mean：')
    print(np.sum(sum)/len(sum))
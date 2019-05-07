from PIR0.neuron_network import *
import csv
import numpy as np

mat = np.zeros((1002, 10 * 1000), dtype=float)
with open("basevaleurs3.csv", "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        mat[i] = [float(a) for a in row]

net = Network([1000, 16, 16, 4])


def f(x):
    ll = [0, 0, 0, 0]
    ll[x - 1] = 1
    return np.array(ll, dtype=float)


training_data = [(mat[:-2, i], f(int(mat[-2, i]))) for i in range(8000)]
test_data = [(mat[:-2, i], int(mat[-2, i])) for i in range(8000, 10000)]

net.SGD(training_data, 40, 50, 2.9, test_data=test_data)

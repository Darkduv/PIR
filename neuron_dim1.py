import neuron_network
import csv
import numpy as np

mat = np.zeros((1002, 10 * 1000), dtype=float)
with open("basevaleurs3.csv", "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        mat[i] = [float(a) for a in row]

net = neuron_network.Network([500, 100, 4])


def f(x):
    ll = [0, 0, 0, 0]
    ll[x-1] = 1
    return np.array(ll, dtype=float)

training_data = [(mat[:500,i], f(int(mat[-2,i]))) for i in range(8000)]+[(mat[500:1000,i], f(int(mat[-2,i]))) for i in range(8000)]
test_data = [(mat[:500,i], f(int(mat[-2,i]))) for i in range(8000, 10000)]

net.SGD(training_data, 40, 10, 2.9, test_data=test_data)

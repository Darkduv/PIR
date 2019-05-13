import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from cmath import phase
phase = np.vectorize(phase)
data = scipy.io.loadmat('base2.mat')

mat = data["Base2"]

def abs_histogram(L, n):
    return np.histogram(abs(L), np.linspace(0, np.max(abs(L)), n))


def circle_histogram(L, n):
    return np.histogram(phase(L), np.linspace(-np.pi, np.pi, n))

def plot_circle(h):
    m=np.sqrt(np.dot(h[0], h[0])/len(h[0]))
    Z = h[0]/m * np.exp(+1j*h[1])[:-1]
    X = Z.real
    Y = Z.imag
    plt.plot(X, Y)

line_test = 1
nn = 10

aa = abs_histogram(mat[:-5, line_test], nn)
plt.hist(aa[0], bins=aa[1])
plt.show()


print(mat[-5:, line_test])
h0 = circle_histogram(mat[:-5, line_test], nn)
plot_circle(h0)

ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.show()



from plot import *
plot_data(mat[:-5, line_test])
nice_plot()
plot_sources(mat[-5:, line_test])

plt.show()

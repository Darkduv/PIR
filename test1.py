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

test = 1
nn = 1000

aa = abs_histogram(mat[:-5,test], nn)
plt.hist(aa[0], bins=aa[1])
plt.show()


print(mat[-5:, test])
h0 = circle_histogram(mat[:-5,test], nn)
plot_circle(h0)

ax = plt.gca()  # gca stands for 'get current axis'
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
plt.show()

def plot_circle0(Z):
    X = Z.real
    Y = Z.imag
    plt.plot(X, Y, "o")


plot_circle0(mat[:-5, test])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
# plt.show()

def make_plot(sources):
    n, qam1, qam2,h1, h2 = sources
    hk = (h1, h2)
    n, qamk = int(n), (int(qam1), int(qam2))
    qam ={4 :np.array([-1-1j,1-1j,1+1j,-1+1j]),
          16:np.array([-1-1j,1-1j,1+1j,-1+1j, -3-3j,3-3j,3+3j,-3+3j,
              -3-1j,3-1j,3+1j,-3+1j, -1-3j,1-3j,1+3j,-1+3j])}
    tab = np.array([0])
    for k in range(1, n+1):
        tab2 = np.array([])
        for t in tab:
            tab2 = np.concatenate((tab2,qam[qamk[k-1]]*hk[k-1]+t))
        tab = tab2
    X = tab.real
    Y = tab.imag
    plt.plot(X, Y, "or")


make_plot(mat[-5:, test])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.show()

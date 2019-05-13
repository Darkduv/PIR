import matplotlib.pyplot as plt
import numpy as np


def nice_plot():
    # init a proper plot design
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))


def plot_data(data_cloud):
    lx = data_cloud.real
    ly = data_cloud.imag
    plt.plot(lx, ly, "o")


def plot_sources(sources):
    n, qam1, qam2, h1, h2 = sources
    hk = (h1, h2)
    n, qam_k = int(n), (int(qam1), int(qam2))
    qam = {4: np.array([-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]),
           16: np.array([-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j, -3 - 3j, 3 - 3j, 3 + 3j, -3 + 3j,
                         -3 - 1j, 3 - 1j, 3 + 1j, -3 + 1j, -1 - 3j, 1 - 3j, 1 + 3j, -1 + 3j])}
    tab = np.array([0])
    for k in range(1, n + 1):
        tab2 = np.array([])
        for t in tab:
            tab2 = np.concatenate((tab2, qam[qam_k[k - 1]] * hk[k - 1] + t))
        tab = tab2
    lx = tab.real
    ly = tab.imag
    plt.plot(lx, ly, "or")

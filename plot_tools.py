import matplotlib.pyplot as plt
import numpy as np


QAM = {4: np.array([-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]),
       16: np.array([-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j, -3 - 3j, 3 - 3j, 3 + 3j, -3 + 3j,
                     -3 - 1j, 3 - 1j, 3 + 1j, -3 + 1j, -1 - 3j, 1 - 3j, 1 + 3j, -1 + 3j])}


def plot_data(data_cloud, col=""):
    """plot a list of complex in the plan"""
    lx = data_cloud.real
    ly = data_cloud.imag
    plt.plot(lx, ly, "o"+col)


def plot_sources(sources):
    """
    plot the modulation grid - constellation diagram-,
    given the characteristics of the sources :
    number (n) and quadrature amplitude modulation (qam)
    """
    n, qam1, qam2, h1, h2 = sources
    hk = (h1, h2)
    n, qam_k = int(n), (int(qam1), int(qam2))
    tab = np.array([0])
    for k in range(1, n + 1):  # we add the points of the qam of the source k on those caused by the sum
        # of the previous k-1 sources. For k = 1, the source doesn't have predecessors.
        # So we add the points to '0'.
        tab = np.concatenate(tuple(QAM[qam_k[k - 1]] * hk[k - 1] + t for t in tab))

    plot_data(tab, col="r")


def nice_plot():
    """
    init a proper plot design
    """
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))


def nice_show():
    nice_plot()
    plt.show()

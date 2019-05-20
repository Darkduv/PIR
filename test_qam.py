from plot_tools import *  # import also matplotlib.pyplot and numpy. no need to re-import them

from bdd_loader import *

from cmath import phase
phase = np.vectorize(phase)  # we vectorize the function to apply it on np.arrays


# mat = load_db("bdd_complex.npy")
mat = load_db("bdd_rotate_20.npy")


def abs_histogram(ll, n):
    """Return histogram of abs(L) over [0, max(|L|)], sampling with $n$ values"""
    return np.histogram(abs(ll), np.linspace(0, np.max(abs(ll)), n))


def plot_histogram(ll, n):
    a = abs_histogram(ll, n)
    plt.hist(a[0], bins=a[1])
    plt.show()


def circle_histogram(ll, n):
    """Return histogram of arg(L) -phase-
    over [-pi, pi], sampling with $n$ values"""
    return np.histogram(phase(ll), np.linspace(-np.pi, np.pi, n))


def plot_circle(h):
    """Plot the histogram h in the complex plan, 'normalized' by its mean"""
    m = np.mean(h[0] ** 2)
    Z = h[0] / m * np.exp(+1j * h[1][:-1])
    X = np.zeros(2*len(Z), dtype=float)
    Y = np.zeros(2*len(Z), dtype=float)
    X[::2] = Z.real
    Y[::2] = Z.imag
    plt.plot(X, Y)


# We test how "looks" an entry. Change line_test to change which entry is plot.

# ------ Test absolute histogram. Not relevant. ------
line_test = 20000
n_samples = 10  # number of values for the sampling

plot_histogram(mat[:-5, line_test], n_samples)


# ------Test histogram of phase. ------
h0 = circle_histogram(mat[:-5, line_test], n_samples)
plot_circle(h0)
nice_show()


# ------ Comparison of the cloud data and the modulation diagram
# if we know the  settings of the sources ------
plot_data(mat[:-5, line_test])
plot_sources(mat[-5:, line_test])
nice_show()

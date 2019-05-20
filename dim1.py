import csv
import numpy as np
import matplotlib.pyplot as plt


# we import the data of the csv in a np.array $mat$
mat = np.zeros((1002, 10 * 1000), dtype=float)
with open("base_values_3.csv", "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        mat[i] = [float(a) for a in row]


# we build a filter in order to smooth the curve of the data
len_sm = 11
SM = np.ones(len_sm, dtype=float)/(len_sm+2)
SM[len_sm//2] *= 4.5   # adjusted over a few examples.


# SM = np.array([1/7, 1/7, 3/7, 1/7, 1/7])
def smooth(tab):
    return np.convolve(tab, SM, "same")


def derivative(tab):
    """
    we compute the derivative of a FFT
    """
    k = np.array(list(range(0, n // 2)) + [0] + list(range(-n // 2 + 1, 0)))
    return 1.0j * k * 2 * np.pi / 36 * tab


# -----------------  TEST over 1 entry ------------------
n_test = 0  # entry we want to test

n = 193
# plot of the histogram, and then the function we associate to the points.
bins = np.linspace(-18, 18, n + 1)
a = np.histogram(mat[:-2, n_test], bins=bins)
plt.bar(a[1][:-1], a[0], 36 / n)
plt.show()

# we compute the derivative of a (smoothed)
a_sm = smooth(smooth(a[0]))  # double smooth appears to be better.
B = derivative(np.fft.fft(a_sm))
C = np.fft.ifft(B)

# we plot the curves.
plt.plot(bins[:-1], smooth(np.real(C)))
plt.plot(bins[:-1], np.real(a_sm))
plt.show()


def count(func):
    """We count the number of time where the function equals 0, and decreasing
    (first > 0, then < 0)"""
    maxis = 0
    for i in range(1, len(func) - 1):
        if func[i - 1] > 0 > func[i + 1]:
            maxis += 1
            # 0 are not exacts, so we have a_i-1 > a_i > 0 > a_i+1 > a_i+2
            # so each root is counted twice.
    return maxis // 2


def automate(i):
    """return the -approximate- number of gaussian curves that were summed up."""
    hist = np.histogram(mat[:-2, i], bins=bins)
    hist_sm = smooth(smooth(hist[0]))
    b = derivative(np.fft.fft(hist_sm))
    c = np.fft.ifft(b)
    c = smooth(np.real(c))
    return count(c)


# test over the given database to see if the algorithm work.
n_ok = 0
for i in range(10 * 1000):
    if automate(i) == mat[-2, i]:
        n_ok += 1
print("Percentage of good answers : ", n_ok / 10000)

import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from PIR0.main_nicolas import *


mat = np.zeros((1002, 10 * 1000), dtype=float)
with open("basevaleurs3.csv", "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        mat[i] = [float(a) for a in row]

sm = np.array([1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 4 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13])


# sm = np.array([1/7, 1/7, 3/7, 1/7, 1/7])
def smooth(tab):
    return np.convolve(tab, sm, "same")


def derivative(tab):
    k = np.asarray(list(range(0, n // 2)) + [0] + list(range(-n // 2 + 1, 0)))
    return 1.0j * k * 2 * np.pi / 36 * tab


n = 193
n_test = 0
bins = np.linspace(-18, 18, n + 1)
a = np.histogram(mat[:-2, n_test], bins=bins)
plt.bar(a[1][:-1], a[0], 36 / n)
plt.show()

a_sm = smooth(smooth(smooth(a[0])))

B = derivative(np.fft.fft(a_sm))
C = np.fft.ifft(B)

plt.plot(bins[:-1], smooth(np.real(C)))
plt.plot(bins[:-1], np.real(a_sm))
plt.show()


def count(func):
    maxis = 0
    for i in range(1, len(func) - 1):
        if func[i - 1] > 0 > func[i + 1]:
            maxis += 1
    return maxis // 2  # 0 are not exacts, so we have a_i-1 > a_i > 0 > a_i+1 > a_i+2


def automate(i):
    hist = np.histogram(mat[:-2, i], bins=bins)
    hist_sm = smooth(smooth(smooth(hist[0])))
    b = derivative(np.fft.fft(hist_sm))
    c = np.fft.ifft(b)
    c = smooth(np.real(c))
    return count(c)


n_ok = 0
for i in range(10 * 1000):
    if automate(i) == mat[-2, i]:
        n_ok += 1
print(n_ok / 10000)

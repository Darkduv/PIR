import numpy as np
import csv


# data = scipy.io.loadmat('base2.mat')

# mat = data["Base2"]

# np.savetxt("bdd.csv",mat,delimiter=",")

def convert_real(x):  # un neurone qui rend le nombre de sources, entre 0 et 4
    answer = [0, 0, 0, 0]
    answer[int(x) - 1] = 1
    return np.array(answer, dtype=float).reshape((4, 1))


def convert_complex(x):  # 6 cas possibles : 1.4 / 1.16 / 2.44 / 2.416 / 2.164 /2.1616
    if x == 1.416 or x == 1.44:
        return np.array([1, 0, 0, 0, 0, 0], dtype="float").reshape((6, 1))
    if x == 1.1616 or x == 1.164:
        return np.array([0, 1, 0, 0, 0, 0], dtype="float").reshape((6, 1))
    if x == 2.44:
        return np.array([0, 0, 1, 0, 0, 0], dtype="float").reshape((6, 1))
    if x == 2.416:
        return np.array([0, 0, 0, 1, 0, 0], dtype="float").reshape((6, 1))
    if x == 2.164:
        return np.array([0, 0, 0, 0, 1, 0], dtype="float").reshape((6, 1))
    if x == 2.1616:
        return np.array([0, 0, 0, 0, 0, 1], dtype="float").reshape((6, 1))
    return np.array([0, 0, 0, 0, 0, 0], dtype="float").reshape((6, 1))


def process_real(db, begin, end):
    H, _ = db.shape
    training_data = []
    for i in range(begin, end):
        training_data.append((db[:-2, i].reshape(H - 2, 1), convert_real(db[-2, i])))
    return np.array(training_data)


def process_complex(db, begin, end):
    H, _ = db.shape
    training_data = []
    for i in range(begin, end):
        number = float(str(int(db[-5, i].real)) + str(".") + str(int(db[-4, i].real)) + str(int(db[-3, i].real)))
        training_data.append(
            (np.concatenate((db[:-5, i].real, db[:-5, i].imag)).reshape((2 * (H - 5), 1)), convert_complex(number)))
    return np.array(training_data)


def load_db(filename):
    return np.load(filename)


def save_db(db):
    np.save("bdd_rotate.npy", db)


def rotate_db_theta(db, theta):
    d = db.copy()
    d[:-5, :] *= np.exp(1j * theta)
    d[-2:, :] *= np.exp(1j * theta)
    return d


def rotate_db(db, n):
    theta_list = np.linspace(0, np.pi * 2, n)
    db_list = []
    for i, theta in enumerate(theta_list):
        db_list.append(rotate_db_theta(db, theta))
        print(100 * i / n, "%")
    return np.concatenate(db_list, axis=1)

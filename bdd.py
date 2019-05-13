import scipy.io
import matplotlib.pyplot as plt
import numpy as np

#data = scipy.io.loadmat('base2.mat')

#mat = data["Base2"]

#np.savetxt("bdd.csv",mat,delimiter=",")

def load_db(filename):
    return np.genfromtxt(filename,delimiter=",",dtype="complex")

def save_db(db):
    np.savetxt("bdd_rotate.csv",db,delimiter=",")

def rotate_db_theta(db,theta):
    d=db.copy()
    d[:-5,:]*=np.exp(1j*theta)
    d[-2:,:]*=np.exp(1j*theta)
    return d

def rotate_db(db,n):
    theta_list=np.linspace(0,np.pi*2,n)
    for i,theta in enumerate(theta_list):
        db = np.concatenate((db,rotate_db_theta(db,theta)),axis=1)
        print(100*i/n,"%")
    return db

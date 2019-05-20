from neuron_network_medium import *
from bdd_loader import *
import numpy as np

EPOCHS = 400
MINI_BATCH_SIZE = 100 
ETA = 20

#/!\ enlever la méthode cost si importation simple

def complex():
	net_complex = Network([2000, 32, 32, 6],cost=QuadraticCost)
	db = load_db("bdd_rotate_20.npy")
	training_data = process_complex(db,0,8000)
	t_data = process_complex(db,8000,10000)
	net_complex.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA,evaluation_data=t_data,learning_rate_schedule=True)
	return net_complex,training_data,t_data

def real():
	#eta de l'ordre de 8
	net_real = Network([1000, 32, 16, 4],cost=QuadraticCost)
	db = load_db("bdd_real.npy")
	training_data = process_real(db,0,8000)
	t_data = process_real(db,8000,10000)
	net_real.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, ETA,evaluation_data=t_data)
	return net_real,training_data,t_data

complex()
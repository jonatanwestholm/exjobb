# testing.py

import numpy as np
import sim
import models


case = "AR"

if case == "AR":
	k = 1
	p = 3
	q = 0
	A = np.array([[0.5,0,0]])
	C = np.array([[1]])
elif case == "MA":
	k = 1
	p = 0
	q = 2
	A = np.array([[]])
	C = np.array([[1, 0.5, -0.2]])
elif case == "ARMA":
	k = 2
	p = 2
	q = 1

	A = np.array([[0.5,-0.2,0,0],[0,0,0.5,-0.2]])
	C = np.array([[1,0],[0,1]])


Y = sim.varma_sim(C,A,500)

#print(Y)

# create VARMA training model

lamb = 0.99

mod = models.VARMA(list(range(k)),[p,q],lamb)

mod.initiate(Y[:10])
mod.learn(Y[10:])
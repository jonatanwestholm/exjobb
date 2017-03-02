# testing.py

import numpy as np
import matplotlib.pyplot as plt

import sim
import models

case = "AR"

if case == "AR":
	k = 1
	p = 3
	q = 0
	A = np.array([[0.5,0,0.2]])
	C = np.array([[1]])
elif case == "MA":
	k = 1
	p = 0
	q = 2
	A = np.array([[]])
	C = np.array([[1, 0.2, -0.2]])
elif case == "ARMA":
	k = 1
	p = 2
	q = 2
	A = np.array([[0.5, -0.2]])
	C = np.array([[1, 0.5, -0.2]])

elif case == "VARMA":
	k = 2
	p = 2
	q = 2

	A = np.array([[0.5,-0.2,0,0],[0,0,0.5,-0.2]])
	C = np.array([[1,0],[0,1]])

N = 1000
Y = sim.varma_sim(C,A,N)

#print(Y)

# create VARMA training model

lamb = 0.97

re = 0.005 # state variance
rw = 50 # output variance
bk = 500

mod = models.VARMA(list(range(k)),[p,q])

#mod.initiate_rls(Y[:10],lamb)
mod.initiate_kalman(re,rw)
A_hist,C_hist = mod.learn(Y[:bk])


#print("\nSwitching hyperparameters!\n")

mod.learners[0].set_variances(0.0001,50)
for i in range(1):
	A_h, C_h = mod.learn(Y[bk:])

	A_hist.extend(A_h)
	C_hist.extend(C_h)


#print(A_hist)

plt.plot(np.array(A_hist)[:,0],'b')
plt.plot(np.array(A_hist)[:,1],'g')
plt.plot(np.array(A_hist)[:,2],'r')
plt.show()
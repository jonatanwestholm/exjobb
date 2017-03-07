# testing.py

import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

import sim
import models

def flatten(lst):
	return [elem for sublist in lst for elem in sublist]

def annealing(data,mod,re_series,rw_series,initiate=False):
	A_hist = np.zeros([1,k,k*p])
	C_hist = np.zeros([1,k,k*q])
	step_length = int(len(data)/len(re_series))
	if initiate:
		mod.initiate_kalman(re_series[0],rw_series[0])
	i = 0
	for re,rw in zip(re_series,rw_series):
		for learner in mod.learners:
			learner.set_variances(re,rw)
		A_h,C_h = mod.learn(data[i*step_length:(i+1)*step_length])
		A_hist = np.concatenate([A_hist,A_h],axis=0)
		C_hist = np.concatenate([C_hist,C_h],axis=0)
		i+=1

	A_hist = A_hist[1:]
	C_hist = C_hist[1:]

	return A_hist,C_hist

def ruminate(data,mod,re_series,rw_series,iterations,meta_series):
	A_hist = np.zeros([1,k,k*p])
	C_hist = np.zeros([1,k,k*q])
	mod.initiate_kalman(re_series[0],rw_series[0])
	for i in range(iterations):
		mod.reset()
		start = 0 #random.randint(0,200)
		A_h,C_h = annealing(data[start:],mod,meta_series[i]*re_series,rw_series)
		A_hist = np.concatenate([A_hist,A_h],axis=0)
		C_hist = np.concatenate([C_hist,C_h],axis=0)

	A_hist = A_hist[1:]
	C_hist = C_hist[1:]		
		
	return A_hist,C_hist

parser = argparse.ArgumentParser()
parser.add_argument('-c','--case',dest = 'case',default="",help='Name of case')
args = parser.parse_args()
case = args.case

if case == "AR":
	A = np.array([[0.5,0,0.2]])
	C = np.array([[1]])

elif case == "MA":
	A = np.array([[]])
	C = np.array([[1, 0.5, -0.2,0.3]])

elif case == "ARMA":
	A = np.array([[0.2001,-0.3001,0.44,0.65,-0.1]])
	C = np.array([[1, 0.5001,0,-0.27]])

elif case == "VAR":
	A = np.array([[0.5001,0.1,0.2],[0.1,0.5001,-0.2],[-0.1,0.2,0.5001]])
	C = np.array([[1,0,0],[0,1,0],[0,0,1]])

N = 2000
Y = sim.varma_sim(C,A,N)

#print(Y)

# create VARMA training model

k = np.shape(A)[0]
p = int(np.shape(A)[1]/k)
q = int(np.shape(C)[1]/k)-1

#lamb = 0.97

re = 0.001# state variance
rw = 500 # output variance
#bk = 500

mod = models.VARMA(list(range(k)),[p,q])

#mod.initiate_rls(Y[:10],lamb)
#mod.initiate_kalman(re,rw)
#A_hist,C_hist = mod.learn(Y)

#print("\nSwitching hyperparameters!\n")

#mod.learners[0].set_variances(0.0001,50)
'''
for i in range(1):
	A_h, C_h = mod.learn(Y[bk:])

	A_hist.extend(A_h)
	C_hist.extend(C_h)
'''

num_series = 100
iterations = int(10000/N)
re_series = np.logspace(-1,-10,num_series)
rw_series = 500*np.logspace(0,-1,num_series)
meta_series = np.logspace(0,0,iterations)
#A_hist,C_hist = annealing(Y,mod,re_series,rw_series,initiate=True)
A_hist,C_hist = ruminate(Y,mod,re_series,rw_series,iterations,meta_series)

#print(np.shape(A_hist[:,,:]))
N = np.shape(A_hist)[0]
if "AR" in case:
	for j in range(k):
		plt.figure()
		plt.title("AR coefficients for y_{0:d}".format(j+1))
		for i in range(k*p):
			plt.plot(A_hist[:,j,i])
		for i in range(k*p):
			plt.plot([0,N],[-A[j][i]]*2)
		legends = flatten([["a({0:d},{1:d})".format(jx+1,ix+1) for ix in range(p)] for jx in range(k)]) + flatten([["a({0:d},{1:d})_gt".format(jx+1,ix+1) for ix in range(p)] for jx in range(k)])
		plt.legend(legends) #["a{0:d}".format(i+1) for i in range(k*p)]+["a{0:d}_gt".format(i+1) for i in range(p)])
if "MA" in case:
	plt.figure()
	plt.title("MA coefficients")
	for i in range(q):
		plt.plot(C_hist[:,0,i])
	for i in range(q):
		plt.plot([0,N],[C[0][i+1]]*2)
	plt.legend(["c{0:d}".format(i+1) for i in range(q)]+["c{0:d}_gt".format(i+1) for i in range(q)])

plt.show()



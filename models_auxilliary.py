# models_auxilliary.py
#[Rodan2011] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2011): 131-144.

import numpy as np
from scipy import sparse

def ESN_A(architecture,N,r=0.5,b=0.05):

	if architecture == "DLR": # Delay Line Reservoir
		arr = np.ones([N-1,1]) #np.random.randint(0,2,[N-1,1])*2 - 1
		arr = r*np.ravel(arr)

		A = sparse.diags(arr,-1)

	elif architecture == "DLRB": # Delay Line Reservoir with Backward Connections
		arr_fwd = np.ones([N-1,1]) #np.random.randint(0,2,[N-1,1])*2 - 1
		arr_back = np.ones([N-1,1]) #np.random.randint(0,2,[N-1,1])*2 - 1

		arr_fwd = r*np.ravel(arr_fwd)
		arr_back = b*np.ravel(arr_back)

		A = sparse.diags([arr_fwd,arr_back],[-1,1])

	elif architecture == "SCR": # Simple Cycle Reservoir
		arr = np.random.randint(0,2,[N,1])*2 - 1

		arr = r*np.ravel(arr[:-1])

		A = sparse.diags([arr,[r]],[-1,N-1])
		

	'''
	elif architecture == "RANDOM":
		num_each = 3

		A = np.zeros([N,N])

		for i in range(N):
			idxs = np.random.choice(N,num_each,replace=False)
			A[i,idxs] = 1

		A = r*A
	'''

	return A

def ESN_B(architecture,M,N,v=1):
	B = np.zeros([N,M])

	if architecture == "UNIFORM":
		for i in range(N):
			B[i,:] = np.random.random([1,M]) < 0.5 #*2 - 1
	elif architecture == "FULL":
		for i in range(N):
			B[i,:] = (np.random.random([1,M]) < 0.5)*2 - 1
	elif architecture == "DIRECT":
		B[:M,:] = np.eye(M)
	elif architecture == "SELECTED":
		arr = np.random.choice(M,N,replace=True)

		for i,elem in enumerate(arr):
			B[i,elem] = (np.random.random() < 0.5)*2-1
	elif architecture == "SECTIONS":
		p = int(N/M)+1

		j = 0
		for i in range(N):
			B[i,j] = (np.random.random() < 0.5)*2-1
			if i%p == p-1:
				j += 1
	elif architecture == "SECTIONS_INIT":
		p = int(N/M)+1

		j = 0
		first = 1
		for i in range(N):
			if first:
				B[i,j] = 1 #(np.random.random() < 0.5)*2-1
				first = 0
			if i%p == p-1:
				j += 1
				first = 1

	B = v*B

	return B

'''
def ESN_C(architecture,N,Oh):
	if architecture == "SELECTED":
		v = 1

		arr = np.random.choice(N,Oh,replace=False)

		C = np.zeros([Oh,N])

		for i,elem in enumerate(arr):
			C[i,elem] = 1

		C = C*v

	return C
'''

def ESN_f(architecture,N,sections=[]):
	if architecture == "LIN":
		f = lambda x: x
	elif architecture == "TANH":
		f = lambda x: np.tanh(x)

	return f

def poslin(thres,bottom,start):
	return lambda x: (x>=thres)*(x+start-bottom) + bottom

THRES_LEVEL = 1000
def make_thres(M,turn_on=True,direct_input=True):
	
	B = THRES_LEVEL*ESN_B("SELECTED",M,1)
	B = np.abs(B)

	if direct_input:
		B += ESN_B() #wait....how to do with non-binary inputs?

class Component:
	def __init__(self,M,N,topology,A_arch="",r=0,B_arch="",v=0,f_arch=""): # f_arch a.k.a. activation
		if topology == "FLEX":
			self.A = [ESN_A(A_arch,r)]
			self.B = [ESN_B(B_arch,v)]
			self.f = [ESN_f(f_arch) for i in range(N)]
		elif topology == "THRES":
			pass
		elif topology == "TRIGGER":
			pass
		elif topology == "HEIGHTSENS":
			pass
		elif topology == "ANOM":
			pass

components = {"AR": ["FLEX","DLR",1,"SECTIONS_INIT",1,"LIN"],
			  "RODAN": ["FLEX","DLR",0.5,"SECTIONS",1,"TANH"],
			  "DIRECT": ["FLEX","DLR",0,"DIRECT",1,"LIN"],
			  "THRES": ["THRES"],
			  "TRIGGER": ["TRIGGER"],
			  "HEIGHTSENS": ["HEIGHTSENS"],
			  "ANOM": ["ANOM"]}

def compound_ESN(spec,M):
	A_comps = []
	B_comps = []
	f_comps = []

	for key in spec:
		comp = Component(M,spec[key],*components[key])
		A_comps += comp.A
		B_comps += comp.B
		f_comps += comp.f

	A = sparse.block_diag(A_comps)
	B = np.concatenate(B_comps,axis=0)
	f = lambda x: np.array([f_comp(x_elem) for x_elem,f_comp in zip(x,f_comps)]).reshape(x.shape)

	return A,B,f









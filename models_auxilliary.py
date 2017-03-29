# models_auxilliary.py
#[Rodan2011] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2011): 131-144.

import numpy as np

def ESN_A(architecture,N):
	if architecture == "DLR": # Delay Line Reservoir
		r = 0.5

		arr = np.random.randint(0,2,[N-1,1])*2 - 1
		arr = r*np.ravel(arr)

		A = np.diag(arr,-1)

	elif architecture == "DLRB": # Delay Line Reservoir with Backward Connections
		r = 0.5
		b = 0.05

		arr_fwd = np.random.randint(0,2,[N-1,1])*2 - 1
		arr_back = np.random.randint(0,2,[N-1,1])*2 - 1

		arr_fwd = r*np.ravel(arr_fwd)
		arr_back = b*np.ravel(arr_back)

		A = np.diag(arr_fwd,-1)

		for i,back in enumerate(arr_back):
			A[i,i+1] = back

	elif architecture == "SCR": # Simple Cycle Reservoir
		r = 0.5

		arr = np.random.randint(0,2,[N,1])*2 - 1

		arr = r*np.ravel(arr[:-1])

		A = np.diag(arr,-1)
		A[0,-1] = arr[-1]		

	return A

def ESN_B(architecture,M,N):
	if architecture == "UNIFORM":
		v = 1

		B = np.zeros([N,M])

		for i in range(N):
			B[i,:] = np.random.randint(0,2,[1,M]) #*2 - 1

		B = B*v

	elif architecture == "DIRECT":
		v = 1

		B = np.zeros([N,M])

		B[:M,:] = np.eye(M)

		B = B*v

	return B

def ESN_C(architecture,N,O):
	if architecture == "SELECTED":
		v = 1

		arr = np.random.choice(N,O,replace=False)

		C = np.zeros([O,N])

		for i,elem in enumerate(arr):
			C[i,elem] = 1

		C = C*v

	return C





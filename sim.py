# sim.py
# simulates different types of data and returns them

import numpy as np 
import scipy.signal as ssignal
import scipy.linalg as slinalg

class LSS:
	def __init__(self,A,B,C):
		self.A = A
		self.B = B
		self.C = C
		self.X = np.zeros([len(A),1])

	def iterate(self,U):
		#self.X = np.dot(self.A,self.X) + np.dot(self.B,U)
		#print(np.shape(self.A))
		#print(np.shape(self.X))
		#update = 
		#print(update)
		#print(np.dot(self.B,U))
		#print(self.A)
		#print(self.X)
		#print(self.B)
		#print(U)
		#print(self.C)
		self.X = np.matmul(self.A,self.X) + np.matmul(self.B,U)
		#print(self.X)
		return np.dot(self.C,self.X)

def arma_sim(C,A,T,num):
	e = np.random.normal(0,1,[num,T])
	return ssignal.filtfilt(C,A,e)

def shift_block(size):
	if size == 0:
		return []
	elif size == 1:
		return np.array([0])
	else:
		return np.diag(np.ones(size-1),-1)

def one_hot(n,k):
	if n == 0:
		return []
	else:
		v = np.zeros([n,1])
		v[k] = 1
		return v

def varma_sim(C,A,T):
	# determine orders
	k = np.shape(C)[0]
	p = int(np.shape(A)[1]/k)
	q = int(np.shape(C)[1]/k)-1

	e = np.random.normal(0,1,[T, k])

	# define LSS
	A_arrays = [shift_block(p) for i in range(k)] + [shift_block(q+1) for i in range(k)]
	LSS_A = slinalg.block_diag(*A_arrays)
	
	'''
	p_hot = np.zeros([p,1])
	p_hot[0] = 1
	q_hot = np.zeros([q,1])
	q_hot[0] = 1
	'''
	B_arrays = [one_hot(p,0) for i in range(k)] + [one_hot(q+1,0) for i in range(k)]
	#print(B_arrays)
	LSS_B = slinalg.block_diag(*B_arrays)

	LSS_C = np.concatenate([-A,C],axis=1)

	#print(LSS_A)
	#print(LSS_B)
	#print(LSS_C)

	lss = LSS(LSS_A,LSS_B,LSS_C)

	# simulate
	y_t = np.zeros([k,1])
	y = np.zeros([T,k])
	for t in range(T):
		e_t = e[t,:].reshape((k,1))
		#print(np.shape(e_t))
		#print(np.shape(y_t))
		if p:
			U = np.concatenate([y_t,e_t])
		else:
			U = e_t
		y_t = lss.iterate(U)
		#print(y_t)
		#print(y[t,:])
		y[t,:] = y_t.reshape((k,)) # this reshape thing is the dumbest

	return y
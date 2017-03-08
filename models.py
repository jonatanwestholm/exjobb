# models.py
# implements hidden-state models for multivariate time series

import numpy as np

def is_tensor(X,order=2):
	if sum(np.array(np.shape(X)) > 1) == order:
		return True
	else:
		return False

class MTS_Model:
	def __init__(self):
		pass

	def reset(self):
		pass

	def update(self,data):
		pass

	def update_array(self,data):
		for dat in data:
			self.update(dat)

	def learn(self,data,gt=[]):
		pass

	def predict(self):
		pass

class Trivial(MTS_Model):

	def __init__(self):
		pass

	def update(self,data):
		self.data = data	

	def predict(self,k):
		return self.data

class VARMA(MTS_Model):

	def __init__(self,orders):
		k = orders[0]
		self.k = k
		self.p = orders[1]
		self.q = orders[2]
		self.A = np.zeros([k,k*self.p])
		self.C = np.zeros([k,k*self.q])
		self.Y = [[0]*k]*self.p
		self.E = [[0]*k]*self.q

	def reset(self):
		self.Y = [[0]*self.k]*self.p
		self.E = [[0]*self.k]*self.q

	def initiate_kalman(self,re,rw):
		N = self.k*(self.p+self.q)
		self.learners = [Kalman(N,re,rw) for i in range(self.k)]

	def initiate_rls(self,Y,lamb):
		self.lamb = lamb
		N = len(Y)
		k = self.k
		p = self.p
		q = self.q
		X = np.random.normal(0,1,[N-p,k*(p+q)])
		#print(X)
		for i in range(N-p):
			X[i,:k*p] = Y[i:(i+p),:][::-1].T.reshape([1,k*p])
		#X = [ for i in range(N-p)]
		self.learners = [RLS(X,Y[p:N,j],self.lamb) for j in range(k)]

		self.Y = [Y[i] for i in range(N-1,N-p-1,-1)]
		#print(self.Y)

	def update(self,y):
		# simplest nontrivial estimation of e(t)
		e = y - self.predict(1)
		#print(e)

		self.E.insert(0,e)
		self.E = self.E[:-1]

		self.Y.insert(0,y)
		self.Y = self.Y[:-1]

	'''
	def make_state(self):
		k = self.k
		p = self.p
		q = self.q
		#print(self.Y)
		#print(self.E)
		if p:
			yresh = np.reshape(self.Y,[1,k*p])
		else:
			yresh = [[]]
		if q:
			eresh = np.reshape(self.E,[1,k*q])
		else:
			eresh = [[]]
		return np.concatenate([yresh,eresh],axis=1)

	def make_transfer(self):
		k = self.k
		p = self.p
		q = self.q
		return np.concatenate(np.reshape(self.A,[1,k*k*p]), np.reshape(self.C,[1,k*k*q]))
	'''

	def make_column(self,data):
		if is_tensor(data[0],1):
			arr = np.concatenate([np.ravel(dat) for dat in data])
			return arr.reshape([len(arr),1])
		else:
			return np.concatenate(data,axis=1)

	# would like to do a proper prediction here with polynomial division and everything
	def predict(self,k):
		if k == 1:
			LSS_C = self.make_column([self.A,self.C]).T
			LSS_X = self.make_column([self.Y,self.E])
			#print(LSS_C)
			return np.dot(LSS_C,LSS_X)
		else:
			print("Haven't done multistep yet!")

	def learn_private(self,y):
		#print(self.Y)
		#print(self.E)
		#print(self.Y + self.E)
		#print(self.Y)
		#print(self.E)
		x = self.make_column([self.Y,self.E])
		for i in range(self.k):
			learner = self.learners[i]
			theta = learner.update(y[i],x.T) 
			#print(self.p)
			#print(learner.theta)
			#print(self.A[i,:])
			#print(self.k)
			#print(i)
			#print(theta)
			if self.p:
				self.A[i,:] = theta[:self.k*self.p].T
			if self.q:
				self.C[i,:] = theta[self.k*self.p:].T
		self.update(y)

		#print("A:")
		#print(self.A)
		#print("C:")
		#print(self.C)
		return self.A,self.C

	def learn(self,Y):
		A_hist = np.zeros([1,self.k,self.k*self.p])
		C_hist = np.zeros([1,self.k,self.k*self.q])
		if is_tensor(Y,0+(self.k>1)):
			self.learn_private(Y.T)				
		else:
			for y in Y:
				A,C = self.learn_private(y.T)
				A_hist = np.concatenate([A_hist,[A]],axis=0)
				C_hist = np.concatenate([C_hist,[C]],axis=0)

		A_hist = A_hist[1:]
		C_hist = C_hist[1:]
		return A_hist,C_hist

	def annealing(self,data,re_series,rw_series,initiate=False):
		k = self.k
		p = self.p
		q = self.q

		A_hist = np.zeros([1,k,k*p])
		C_hist = np.zeros([1,k,k*q])
		step_length = int(len(data)/len(re_series))
		if initiate:
			self.initiate_kalman(re_series[0],rw_series[0])
		i = 0
		for re,rw in zip(re_series,rw_series):
			for learner in self.learners:
				learner.set_variances(re,rw)
			A_h,C_h = self.learn(data[i*step_length:(i+1)*step_length])
			A_hist = np.concatenate([A_hist,A_h],axis=0)
			C_hist = np.concatenate([C_hist,C_h],axis=0)
			i+=1

		A_hist = A_hist[1:]
		C_hist = C_hist[1:]

		return A_hist,C_hist

	def ruminate(self,data,re_series,rw_series,iterations,meta_series):
		k = self.k
		p = self.p
		q = self.q
		
		A_hist = np.zeros([1,k,k*p])
		C_hist = np.zeros([1,k,k*q])
		self.initiate_kalman(re_series[0],rw_series[0])
		for i in range(iterations):
			self.reset()
			start = 0 #random.randint(0,200)
			A_h,C_h = self.annealing(data[start:],meta_series[i]*re_series,rw_series)
			A_hist = np.concatenate([A_hist,A_h],axis=0)
			C_hist = np.concatenate([C_hist,C_h],axis=0)

		A_hist = A_hist[1:]
		C_hist = C_hist[1:]		
			
		return A_hist,C_hist

# helps with learning
# an object whose states is the weights of other models
class RLS:
	def __init__(self,X,Y,lamb):
		self.lamb = lamb

		# solve first as if they were instant
		#print(np.shape(X))
		self.dim = len(X[0])
		#print(self.dim)
		Y = Y.reshape([len(Y),1])
		#print(np.shape(Y))

		#print(X)
		#print(Y)

		self.theta = np.linalg.lstsq(X,Y)[0]
		#print(self.theta)
		#print(X)
		#print(np.dot(X.T,X))
		self.P = np.linalg.inv(np.dot(X.T,X))

	def update_private(self,x,y,lamb=0):
		#print(x)
		x = x.reshape([self.dim,1])
		if not lamb:
			lamb = self.lamb
		#print(self.P)
		#print(x)
		M = np.dot(self.P,x)
		denominator = (lamb + np.dot(x.T,M))
		K = M/denominator

		#print(y)
		#print(x.T)
		#print(self.theta)
		err = y-np.dot(x.T,self.theta)
		#print(K)
		#print(err)
		self.theta += K*err
		self.P = (self.P - np.dot(M,M.T)/denominator)/lamb

		return self.theta

	def update(self,Y,X,block=True):
		if block:
			lamb = 1
		else:
			lamb = self.lamb
		if is_tensor(Y,order=2):
			self.update_private(X[0],Y[0]) # first forgetting round
			for x,y in zip(X[1:],Y[1:]):
				self.update_private(x,y,lamb) # then
		else:
			#print(X)
			self.update_private(X,Y,lamb)

		return self.theta

class Kalman:
	def __init__(self,N,re,rw,A=[],C=[],X=[]):
		if not A:
			A = np.eye(N)
		if not C:
			C = np.zeros([1,N])
		if not X:
			X = np.zeros([N,1])
		
		self.N = N

		self.A = A
		self.C = C
		self.X = X

		self.Re = self.init_r(re,N)
		self.Rw = self.init_r(rw,1)

		self.Rxx = self.Re
		self.Ryy = self.Rw

	def init_r(self,r,N):
		if is_tensor(r,0):
			R = np.eye(N)*r
		elif is_tensor(r,1):
			R = np.diag(r)
		elif is_tensor(r,2):
			R = r
		return R

	def set_variances(self,re,rw):
		self.Re = self.init_r(re,self.N)
		self.Rw = self.init_r(rw,1)		

	def update(self,y,C=[],A=[]):
		if C == []:
			C = self.C
		if A == []:
			A = self.A

		# inference
		#print(self.X)
		K = np.dot(np.dot(self.Rxx,C.T),np.linalg.inv(self.Ryy))
		self.X = self.X + np.dot(K,y-np.dot(C,self.X))
		#print(K)
		#print(y)

		# update
		eye = np.eye(self.N)
		Rxx1 = np.dot(eye-np.dot(K,C),self.Rxx)
		self.Rxx = np.dot(self.A,np.dot(Rxx1,self.A.T)) + self.Re
		self.Ryy = (np.dot(self.C,np.dot(Rxx1,self.C.T)) + self.Rw).astype(dtype='float64') 

		return self.X
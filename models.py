# models.py
# implements hidden-state models for multivariate time series

import numpy as np

def is_matrix(X):
	if sum(np.array(np.shape(X)) > 1) > 1:
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

	def __init__(self,subgroup):
		self.subgroup = subgroup

	def update(self,data):
		self.data = data	

	def predict(self,k):
		return self.data

class VARMA(MTS_Model):

	def __init__(self,subgroup,orders,lamb):
		self.subgroup = subgroup
		k = len(subgroup)
		self.k = k
		self.p = orders[0]
		self.q = orders[1]
		self.A = np.zeros([k,k*self.p])
		self.C = np.zeros([k,k*self.q])
		self.Y = [[0]*self.p]*k
		self.E = [[0]*self.q]*k
		self.lamb = lamb

		print(self.A)

	def initiate(self,Y):
		N = len(Y)
		k = self.k
		p = self.p
		q = self.q
		X = np.zeros([N-p,k*(p+q)])
		for i in range(N-p):
			X[i,:k*p] = Y[i:(i+p),:][::-1].T.reshape([1,k*p])
		#X = [ for i in range(N-p)]
		self.learners = [RLS(X,Y[p:N,j],self.lamb) for j in range(k)]

	def update(self,y):
		# simplest nontrivial estimation of e(t)
		e = y - self.predict(1)

		self.E.insert(0,e)
		self.E = self.E[:-1]

		self.Y.insert(0,y)
		self.Y = self.Y[:-1]

	# would like to do a proper prediction here with polynomial division and everything
	def predict(self,k):
		if k == 1:
			LSS_C = np.concatenate([self.A,self.C],axis=1)
			LSS_X = np.concatenate(self.Y + self.E)
			return np.matmul(LSS_C,LSS_X)
		else:
			print("Haven't done multistep yet!")

	def learn_private(self,y):
		#print(self.Y)
		#print(self.Y + self.E)
		x = np.concatenate(self.Y + self.E)
		for i in range(self.k):
			learner = self.learners[i]
			learner.update(x,y[i]) 
			#print(self.p)
			#print(learner.theta)
			#print(self.A[i,:])
			#print(self.k)
			#print(i)
			if self.p:
				self.A[i,:] = learner.theta[:self.p].T
			if self.q:
				self.C[i,:] = learner.theta[self.p:]
		self.update(y)
		print(self.A)

	def learn(self,Y):
		if np.shape(Y)[0] > 1:
			for y in Y:
				self.learn_private(y.T)
		else:
			self.learn_private(Y.T)	

# helps with learning
# an object whose states is the weights of other models
class RLS:
	def __init__(self,X,Y,lamb):
		self.lamb = lamb

		# solve first as if they were instant
		#print(np.shape(X))
		self.dim = len(X[0])
		Y = Y.reshape([len(Y),1])
		#print(np.shape(Y))

		#print(X)
		#print(Y)

		self.theta = np.linalg.lstsq(X,Y)[0]
		#print(self.theta)
		self.P = np.linalg.inv(np.dot(X.T,X))

	def update_private(self,x,y,lamb=0):
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

	def update(self,X,Y,block=True):
		if block:
			lamb = 1
		else:
			lamb = self.lamb
		if is_matrix(Y):
			self.update_private(X[0],Y[0]) # first forgetting round
			for x,y in zip(X[1:],Y[1:]):
				self.update_private(x,y,lamb) # then
		else:
			self.update_private(X,Y,lamb)

		return self.theta

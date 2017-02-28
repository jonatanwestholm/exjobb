# models.py
# implements hidden-state models for multivariate time series

import numpy as np

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

	def __init__(self,subgroup,orders):
		self.subgroup = subgroup
		self.k = len(subgroup)
		self.p = orders[0]
		self.q = orders[1]
		self.A = np.zeros([k,k*self.p])
		self.A = np.zeros([k,k*self.q])
		self.Y = [[0]*orders[0]]*k
		self.E = [[0]*orders[1]]*k

	def initiate(self,Y):
		# bunch of shifts here and then call RLS
		# ...
		# self.learner = RLS(X,Y,lamb)
		pass

	def update(self,y):
		# simplest nontrivial estimation of e(t)
		e = y - self.predict(1)

		self.E.insert(0,e)
		self.E = self.E[:-1]

		self.Y.insert(0,y)
		self.Y = self.Y[:-1]

	# would like to do a proper prediction here with polynomial division and everything
	def predict(self,k):
		pass

	def learn(self,y):
		x = np.array(self.Y + self.E)
		for i in range(self.k):
			learner = self.learners[i]
			learner.update(x,y[i]) #oooh shit, I will need one RLS per feature! should do something to avoid that
			self.A[i,:] = learner.theta[:self.p]
			self.C[i,:] = learner.theta[self.p:]
		self.update(y)

class RLS:
	def __init__(self,X,Y,lamb):
		self.lamb = lamb

		# solve first as if they were instant
		self.theta = np.linalg.solve(X,Y)
		self.P = np.linalg.inv(np.dot(X.T,X))

	def update_private(self,x,y,lamb=0):
		if not lamb:
			lamb = self.lamb
		M = np.dot(self.P,x)
		denominator = (lamb + np.dot(x.T,M))
		K = M/denominator

		self.theta += np.dot(K,y-np.dot(x.T,self.theta))
		self.P = (self.P - np.dot(M,M.T)/denominator)/lamb

		return self.theta

	def update(self,X,Y,block):
		self.update_private(X[0],Y[0]) # first forgetting round
		if block:
			lamb = 1
		else:
			lamb = self.lamb
		for x,y in zip(X[1:],Y[1:]):
			self.update_private(x,y,lamb) # then

		return self.theta

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

	def learn(self,data,gt):
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
		self.A = [0]*orders[0]
		self.C = [0]*orders[1]
		self.Y = [0]*(orders[0]+1)
		self.E = [0]*(orders[1]+1)

	def update(self,y):
		# simplest nontrivial estimation of e(t)
		e = y - self.predict(1)

		self.E.insert(0,e)
		self.E = self.E[:-1]

		self.Y.insert(0,y)
		self.Y = self.Y[:-1]

	def predict(self,k):
		pass
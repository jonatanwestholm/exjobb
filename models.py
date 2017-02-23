# models.py

class MTS_Model:
	def __init__(self):
		pass

	def reset(self):
		pass

	def update(self,data):
		pass

	def learn(self,data,gt):
		pass

	def predict(self):
		pass

class Trivial(MTS_Model):

	def __init__(self,subgroup):
		self.subgroup = subgroup

	def update(self,data):
		if len(data) > 1:
			for dat in data:
				self.update(dat)
		else:
			self.data = data	

	def predict(self,k):
		return self.data

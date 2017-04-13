# models.py
# implements hidden-state models for multivariate time series

import numpy as np
import copy
import time
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import models_auxilliary as mod_aux

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

	def learn(self,Y,label=[]):
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
		self.learners = [mod_aux.Kalman(N,re,rw) for i in range(self.k)]

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
		#print(type(self.Y[0]))
		# simplest nontrivial estimation of e(t)
		#print(self.E)
		e = y - self.predict(1)
		#print(e)

		self.E.insert(0,e)
		self.E = self.E[:-1]

		#print(self.E)

		self.Y.insert(0,y)
		self.Y = self.Y[:-1]

		#print(type(self.Y[0]))

	def make_column(self,data):
		#if is_tensor(data[0],1) or is_tensor(data[0],0):
		arr = np.concatenate([np.ravel(dat) for dat in data])
		return arr.reshape([len(arr),1])
		#else:
		#	return np.concatenate(data,axis=1)

	def make_block(self,data):
		return np.concatenate(data,axis=1)

	# would like to do a proper prediction here with polynomial division and everything
	def predict(self,k,protected=True):
		if not self.k:
			return
		if k == 1:
			LSS_C = self.make_block([self.A,self.C])
			LSS_X = self.make_column([self.Y,self.E])
			return np.dot(LSS_C,LSS_X)
		else:
			# wasteful indeed but simple and correct according to book (Jakobsson2015, 8.148)
			if protected:
				other = copy.deepcopy(self)
				return other.predict(k,protected=False)
			else:
				y = self.predict(1).T[0]
				#print(np.shape(y))
				self.update(y) 
				return self.predict(k-1,protected=False)

	def learn_private(self,y):
		if self.q:
			x = self.make_column([self.Y,self.E])
		else:
			x = self.make_column([self.Y])
			#print(x.T)
		for i in range(self.k):
			learner = self.learners[i]
			theta = learner.update(y[i],x.T) 
			if self.p:
				self.A[i,:] = theta[:self.k*self.p].T
			if self.q:
				self.C[i,:] = theta[self.k*self.p:].T
		self.update(y)

		return self.A,self.C

	def learn(self,Y):
		A_hist = np.zeros([1,self.k,self.k*self.p])
		C_hist = np.zeros([1,self.k,self.k*self.q])
		#print("tensor level: "+ str(0+(self.k>1)))
		#print(Y)
		if mod_aux.is_tensor(Y,0+(self.k>1)):
			self.learn_private(Y.T)				
		else:
			if mod_aux.is_tensor(Y,1):
				Y = self.make_column(Y)
				#print(Y)
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

		#A_hist = A_hist[1:]
		#C_hist = C_hist[1:]

		return A_hist,C_hist

	def set_A_C(self,A,C):
		self.A = A
		self.C = C
		for i in range(self.k):
			learner = self.learners[i]
			if self.k > 1:
				A_row = A[i,:]
				C_row = C[i,:]
			else:
				A_row = A
				C_row = C
			if self.q:
				learner.set_X(self.make_block([A_row,C_row]))
			else:
				learner.set_X(A_row)

class ESN(MTS_Model):

	def __init__(self,purpose,orders,spec,mixing):
		self.purpose = purpose
		size_in,size_out,size_label = orders
		self.M = size_in
		self.Oh = size_out
		self.L = size_label
		
		self.reservoir = mod_aux.Reservoir(self.M,spec,mixing,False)
		self.A,self.B,self.f,self.idx_groups = self.reservoir.get_matrices()
		self.reservoir.print_reservoir()
		#for row in self.A.toarray():
		#	print(row)
		#print(self.B)

		self.N = self.reservoir.total_size()
		
		self.inputs = []
		self.outputs = []

		self.learners = None

		self.reset()

	def reset(self):
		self.X = np.zeros([self.N,1])

	def activate(self,X):
		Y = np.zeros_like(X)
		for f_comp,idx_group in zip(self.f,self.idx_groups):
			start = idx_group[0]
			end = idx_group[1]
			Y[start:end] = f_comp(X[start:end])

		#print(X)
		#print(Y)
		return Y

	def update_private(self,U,X,iterations=1,noise=True):
		for i in range(iterations):
			'''
			print(self.B)
			print(self.B.shape)
			print(U.shape)
			print(self.A_array)
			print(self.A.shape)
			print(X.shape)
			'''
			X = self.activate(self.A*X + np.dot(self.B,U))
			'''
			if noise:
				X = self.activate(self.A*X + np.dot(self.B,U))	
			else:
				X = self.activate(self.A*X + np.dot(self.B,U))
			'''

		return X

	def update(self,Y):
		if mod_aux.is_tensor(Y,int(self.M > 1)):
			U = np.reshape(Y,[self.M,1])
			self.X = self.update_private(U,self.X)
		else:
			for row in Y:
				U = np.reshape(row,[self.M,1])
				self.X = self.update_private(U,self.X)

		return self.X

	def train_Cs(self,X):
		__,S,V = np.linalg.svd(X,full_matrices=False)
		
		S_energy = np.cumsum(S)
		S_energy = S_energy/S_energy[-1]
		plt.plot(S_energy)
		plt.show()
		
		self.Cs = V[:self.Oh,:]
		#self.Cs = np.eye(self.Oh)

	def train_Cw(self,Xs,Y,tikho):
		XX = np.dot(Xs.T,Xs)
		#print(XX.shape)
		XX += np.eye(self.Oh)*tikho**2
		#print(XX)
		XY = np.dot(Xs.T,Y)
		#print(XY)
		self.Cw = np.linalg.lstsq(XX,XY)[0].T
		#print(self.Cw)
		#return np.reshape(self.Cw.T,[1]+list(self.Cw.T.shape))

	def charge(self,U,y=[],burn_in=0):
		self.reset()
		U_burn_in = U[:burn_in]
		U = U[burn_in:]
		for u in U_burn_in:
			self.update(u)

		T = U.shape[0]
		X_vec = np.zeros([T,self.N])

		for i,u in enumerate(U[:-1]):
			X = self.update(u)
			X_vec[i,:] = np.ravel(X)

		if y == []:
			X_vec = X_vec[:-1,:]
			Y_vec = U[1:,:]
		else:
			Y_vec = y

		self.inputs.append(X_vec)
		self.outputs.append(Y_vec)

	def train(self,tikho):
		X = np.concatenate(self.inputs,axis=0)
		#print(X.shape)
		Y = np.concatenate(self.outputs,axis=0)
		#print(Y.shape)
		#print(X[:5,:])
		#print(X[-5:,:])
		
		self.train_Cs(X)
		Xs = np.dot(X,self.Cs.T)
		self.train_Cw(Xs,Y,tikho)

		print(sorted(np.dot(self.Cw,self.Cs)[0]))

	def out(self,X=[]):
		if X == []:
			X = self.X
		Xs = np.dot(self.Cs,X)

		return np.dot(self.Cw,Xs)

	def predict(self,k,U=[]):
		if self.purpose == "CLASSIFICATION":
			#U = np.zeros([self.M,1])
			if mod_aux.is_tensor(U,self.M>1):
				X = self.update_private(U,self.X,k-1,noise=False)
				y = self.out(X)
			else:
				# note that the prediction changes the state 
				y = []
				for u in U:
					y.append(self.predict(k,u))
					self.update(u)
				y = np.concatenate(y,axis=0)
		elif self.purpose == "PREDICTION":
			X = self.X
			for i in range(k-1):
				y = self.out(X)
				X = self.update_private(y,X,noise=False)
			y = self.out(X)

		return y

class SVM_TS:
	def __init__(self,subgroup,pos_w,style):
		self.subgroup = subgroup
		self.pos_w = pos_w
		self.style = style
		self.reset()

	def reset(self):
		self.X = np.array([[]]*len(self.subgroup)).T
		self.y = np.array([[]]).T
		self.w = np.array([[]]).T 
		if self.style == "SVC":
			self.sv = svm.SVC()
		elif self.style == "SVR":
			self.sv = svm.SVR()
		elif self.style == "MLP":
			self.sv = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    				hidden_layer_sizes=(10,3), random_state=1)

	def charge(self,X,y):
		if self.style == "SVC":
			w = np.ones_like(y)
			w[y==1] = self.pos_w
		elif self.style == "SVR":
			w = 1/(y+1/self.pos_w)
		else:
			w = [[0]]

		self.X = np.concatenate([self.X,X])
		self.y = np.concatenate([self.y,y])
		self.w = np.concatenate([self.w,w])

	def train(self,return_score=False):
		if "SV" in self.style:
			self.sv.fit(self.X,np.ravel(self.y),np.ravel(self.w))
		else:
			self.sv.fit(self.X,np.ravel(self.y))
		if return_score:
			return self.sv.score(self.X,np.ravel(self.y),np.ravel(self.w))

	def predict(self,dat):
		return self.sv.predict(dat)
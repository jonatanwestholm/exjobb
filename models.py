# models.py
# implements hidden-state models for multivariate time series

import numpy as np
import copy
import time
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

import models_auxilliary as mod_aux

def is_tensor(X,order=2):
	#print(X)
	#print(np.shape(X))
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
		if is_tensor(Y,0+(self.k>1)):
			self.learn_private(Y.T)				
		else:
			if is_tensor(Y,1):
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
	def __init__(self,style,orders,architectures):
		self.style = style
		size_in,size_nodes,size_out,size_label = orders
		self.M = size_in
		self.N = size_nodes
		self.Oh = size_out
		self.L = size_label
		
		self.A = mod_aux.ESN_A(architectures[0],self.N)
		self.B = mod_aux.ESN_B(architectures[1],self.M,self.N)
		#self.Cs = mod_aux.ESN_C(architectures[2],self.N,self.Oh)
		self.f = mod_aux.ESN_f(architectures[3],self.N)
		self.f_noise = lambda x: self.f(x) + np.random.normal(0,0.1,x.shape)#1/(1+np.exp(-x))

		self.learners = None
		self.first = 1

		self.reset()

	def reset(self):
		self.X = np.zeros([self.N,1])

	def initiate_kalman(self,re,rw):
		self.learners = [Kalman(self.Oh,re,rw) for i in range(self.L)]

	def update_private(self,U,X,iterations=1,noise=True):
		for i in range(iterations):
			if noise:
				X = self.f_noise(self.A*X + np.dot(self.B,U))	
			else:
				X = self.f(self.A*X + np.dot(self.B,U))

		return X

	def update(self,Y):
		if is_tensor(Y,int(self.M > 1)):
			U = np.reshape(Y,[self.M,1])
			self.X = self.update_private(U,self.X)
		else:
			for row in Y:
				U = np.reshape(row,[self.M,1])
				self.X = self.update_private(U,self.X)

		return self.X

	def make_Cw(self):
		Cw = np.zeros([self.L,self.Oh])
		for i,learner in enumerate(self.learners):
			Cw[i,:] = np.reshape(learner.X,[1,self.Oh])

		self.Cw = Cw
		return Cw

	def set_Cw(self,Cw):
		self.Cw = Cw
		if self.learners:
			for i,learner in enumerate(self.learners):
				learner.X = np.reshape(Cw[i,:],[self.Oh,1])

	def learn_private(self,y,label=None):
		Ys = np.dot(self.Cs,self.X)
		Ys = np.reshape(Ys,[1,self.Oh])

		if not label:
			label = y

		for learner,lab in zip(self.learners,label):
			learner.update(lab,C=Ys)

		#print("prediction: {0:.3f}, gt: {1:.3f}".format(self.predict(1)[0][0],y[0]))
		self.update(y)

		return self.make_Cw()

	def learn(self,Y,labels=[]):
		C_hist = np.zeros([1,self.L,self.Oh])

		if not labels:
			for y in Y:
				C_h = self.learn_private(y)
				C_hist = np.concatenate([C_hist,[C_h]],axis=0)				
		else:
			for y,label in zip(Y,labels):
				C_h = self.learn_private(y,label)
				C_hist = np.concatenate([C_hist,[C_h]],axis=0)

		C_hist = C_hist[1:]

		return C_hist

	def learn_batch(self,Y,labels=[],tikho=0):
		T = Y.shape[0]

		X_vec = np.zeros([T,self.N])

		for i,y in enumerate(Y[:-1]):
			X = self.update(y)
			X_vec[i,:] = np.ravel(X)

		if not labels:
			X_vec = X_vec[:-1,:]
			Y_vec = Y[1:,:]
		else:
			Y_vec = labels

		if self.first:
			U,S,V = np.linalg.svd(X_vec)
			self.Cs = V[:self.Oh,:]
			print(self.Cs.shape)
			#self.Cs = np.eye(self.Oh)
			'''
			S_energy = np.cumsum(S)
			S_energy = S_energy/S_energy[-1]
			plt.plot(S_energy)
			plt.show()
			'''
			self.first = 0
		else:
			pass
			'''
			U,S,V = np.linalg.svd(X_vec)
			U = U[:,:self.N]

			U_auto1 = np.diag(np.dot(self.U.T,self.U))
			U_auto2 = np.diag(np.dot(U.T,U))
			U_cross = np.diag(np.dot(self.U.T,U))

			U_rel = U_cross/np.sqrt(U_auto1*U_auto2)

			plt.plot(U_rel)
			plt.show()
			'''

		X_vec = np.dot(X_vec,self.Cs.T)

		#print(X_vec)

		XX = np.dot(X_vec.T,X_vec)
		#print(XX.shape)
		XX += np.eye(self.Oh)*tikho**2
		#print(XX)
		XY = np.dot(X_vec.T,Y_vec)
		#print(XY)
		self.Cw = np.linalg.lstsq(XX,XY)[0]
		return np.reshape(self.Cw.T,[1]+list(self.Cw.T.shape))

	def out(self,X=[]):
		if X == []:
			X = self.X
		Ys = np.dot(self.Cs,X)

		return np.dot(self.Cw,Ys)

	def predict(self,k):
		if self.style == "CLASSIFICATION":
			U = np.zeros([self.M,1])
			X = self.update_private(U,self.X,k-1,noise=False)
		elif self.style == "PREDICTION":
			X = self.X
			for i in range(k-1):
				y = self.out(X)
				X = self.update_private(y,X,noise=False)
		return self.out(X)

	def annealing(self,dat,re_series,rw_series,initiate=False):
		C_hist = np.zeros([1,self.L,self.Oh])

		step_length = int(len(dat)/len(re_series))
		if initiate:
			self.initiate_kalman(re_series[0],rw_series[0])
		i = 0
		for re,rw in zip(re_series,rw_series):
			for learner in self.learners:
				learner.set_variances(re,rw)
			C_h = self.learn(dat[i*step_length:(i+1)*step_length])
			#print(C_h.shape)
			C_hist = np.concatenate([C_hist,C_h],axis=0)
			i+=1

		C_hist = C_hist[1:]

		return C_hist

	def print_esn_line(self,idx):
		line = ""
		line += " ".join(["{0:.0f}".format(elem).rjust(3,' ') for elem in self.B[idx,:]])
		line += "    |"
		line += " v {0:.2f}".format(self.A[(idx+1)%self.N,idx]).ljust(9,' ')
		line += "| {0:.2f}".format(self.A[(idx-1)%self.N,idx]).ljust(9,' ') + "^"
		line += "|    "
		#out_node = np.where(self.Cs[:,idx] != 0)[0]
		#if len(out_node):
		#	out_node = out_node[0]
		line += " ".join(["{0:.3f}".format(elem).rjust(8,' ') for elem in self.Cs[:,idx]])

		return line

	def print_esn(self):
		esn_print = "\n".join([self.print_esn_line(idx) for idx in range(self.N)])
		print(esn_print)

# helps with learning
# an object whose states is the weights of other models
class Kalman:
	def __init__(self,N,re,rw,A=[],C=[],X=[]):
		if not A:
			A = np.eye(N)
		if not C:
			C = np.zeros([1,N])
		if not X:
			#X = np.zeros([N,1])
			X = np.random.normal(0,0.1,[N,1])
		
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
		
		if np.linalg.norm(self.X) > 10:
			self.X = np.zeros([self.N,1])
			self.Rxx = self.Re
			self.Ryy = self.Rw
			print("reset X")
			#print(self.X)
			#print(self.Re)
			#print(self.Rw)
			#time.sleep(0.5)

		#degeneration step
		#self.X = 0.985*self.X

		return self.X

	def set_X(self,X):
		self.X = X

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

	def update(self,X,y):
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
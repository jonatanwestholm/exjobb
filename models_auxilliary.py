# models_auxilliary.py
#[Rodan2011] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2011): 131-144.

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

esn_component_sizes = {"VAR":1,"RODAN":1,"DIRECT":1,"THRES":2,"TRIGGER":3}

THRES_HIGH = 100000
TH = THRES_HIGH

# turns list of lists into list
def flatten(lst):
	return [elem for sublist in lst for elem in sublist]

def is_tensor(X,order=2):
	#print(X)
	#print(np.shape(X))
	if sum(np.array(np.shape(X)) > 1) == order:
		return True
	else:
		return False

def significant_nodes(X,Y):
	Y = Y-np.min(Y)
	Y = Y/np.max(Y)
	sig = []
	for feat in X.T:
		pos = feat[np.where(Y==1)[0]]
		neg = feat[np.where(Y==0)[0]]

		sig.append(significance_test(pos,neg))

	sig = np.array(sig)
	#print(sig)
	return sig

def significance_test(x1,x2):
	mean1 = np.mean(x1)
	mean2 = np.mean(x2)

	std1 = np.std(x1)
	std2 = np.std(x2)

	return overlap([mean1-2*std1,mean1+2*std1],[mean2-2*std2,mean2+2*std2])

def overlap(int1,int2):
	#print(int1)
	#print(int2)

	if int1[0] > int2[0]:
		tmp = int1
		int1 = int2
		int2 = tmp

	#print(int1)
	#print(int2)

	if int1[0] > int2[0]:
		print(int1)
		print(int2)
		raise

	if int1[1] > int2[1]:
		return 1

	if int1[1] < int2[0]:
		return 0

	ltot = int2[1]-int1[0]

	if ltot == 0:
		return 1

	common = int1[1]-int2[0]

	return common/ltot

def cumulative_singular_values(S,plot=False):
	S_energy = np.cumsum(S)
	S_energy = S_energy/S_energy[-1]
	if plot:
		plt.plot(S_energy)
		plt.title("Cumulative singular values")
		plt.xlabel("Singular value")
		plt.ylabel("Cumulative relative singular values")
		plt.show()

	return S_energy

def ESN_A(architecture,N,r=0.5,b=0.05):

	if architecture == "DLR": # Delay Line Reservoir
		if N > 1:
			arr = np.ones([N-1,1]) #np.random.randint(0,2,[N-1,1])*2 - 1
			arr = r*np.ravel(arr)

			A = sparse.diags(arr,-1)
		else:
			A = sparse.diags(0,0)

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
	elif architecture == "DIAGONAL":
		arr = np.ones([N,1])
		arr = r*np.ravel(arr)

		A = sparse.diags([arr],[0])
	'''
	elif architecture == "DIRECT":
		arr = np.ones([N,])

		A = sparse.diags(arr,0)		
	'''
	'''
	elif architecture == "RANDOM":
		num_each = 3

		A = np.zeros([N,N])

		for i in range(N):
			idxs = np.random.choice(N,num_each,replace=False)
			A[i,idxs] = 1

		A = r*A
	'''
	A = A.tocsr()

	return A

def ESN_B(architecture,M,N,v=1,replace=True,external_input=0):
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
		arr = np.random.choice(M,N,replace=replace)

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
	elif architecture == "SINGLE":
		B[0,external_input] = 1

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

def ESN_f(architecture,thres=0):
	if architecture == "LIN":
		f = lambda x: x
	elif architecture == "TANH":
		f = lambda x: np.tanh(x)
	elif architecture == "POSLIN":
		f = lambda x: (x>thres)*(x-thres)
	elif architecture == "DOUBLE_POSLIN":
		f = lambda x: (np.abs(x)>thres)*x + (np.abs(x)<=thres)*thres
	elif architecture == "THRES":
		f = lambda x: (x>thres)*1.0
	elif architecture == "INVERSE_DECAY":
		f = lambda x: x - thres**2/x

	return f

def make_thres(M, N, direct_input, random_thres, turn_on):
	B = ESN_B("SELECTED",M,2*N)
	B[N:,:] = 0
	if direct_input:
		B[:N,:] = 0

	lower_diag = np.ravel(TH*np.ones([N,1]))

	A = sparse.diags([lower_diag], [-N])

	if random_thres:
		THRES = np.random.normal(0,1,[N,1])
	else:
		THRES = 0

	if turn_on:
		f = [ESN_f("THRES",THRES),ESN_f("POSLIN",THRES_HIGH)]
	else:
		A = -A
		f = [ESN_f("THRES",THRES),ESN_f("POSLIN",0)]

	A = A.tocsr()
	return A,B,f

def make_trigger(M, N, direct_input, random_thres, turn_on):
	B = ESN_B("SELECTED",M,3*N)
	B[N:,:] = 0
	if not direct_input:
		B[:N,0] = 0

	if random_thres:
		THRES = np.random.normal(0,1,[N,1])
	else:
		THRES = np.zeros([N,1])

	A2A = np.zeros([N,1])
	A2B = np.ones([N,1])
	B2B = TH*np.ones([N,1])
	B2C = TH*np.ones([N,1])
	main_diag = np.ravel(np.concatenate([A2A,B2B,A2A]))
	lower_diag = np.ravel(np.concatenate([A2B,B2C]))

	if turn_on:
		f = [ESN_f("THRES",THRES), ESN_f("THRES",0), ESN_f("POSLIN",THRES_HIGH)]
		A = sparse.diags([main_diag,lower_diag],[0,-N])
	else:
		f = [ESN_f("THRES",THRES), ESN_f("THRES",0), ESN_f("POSLIN",0)]
		A = sparse.diags([main_diag,-lower_diag],[0,-N])

	A = A.tocsr()
	return A,B,f

class Node:
	def __init__(self,input_idx,output_idx):
		self.input_idx = input_idx
		self.output_idx = output_idx
		self.inputs = []
		self.outputs = []

	#def set_external_input(self,external,weight):
	#	self.external_inputs.append((external,weight))

	def set_input(self,other,weight):
		self.inputs.append((other,weight))

	def set_output(self,other,weight):
		self.outputs.append((other,weight))

	def backpropagate(self):
		pass

class Component:
	def __init__(self,N): # f_arch a.k.a. activation
		self.N = N
		self.internal_input = []

	def get_matrices(self):
		return [self.A],[self.B],self.f

	def get_typename(self):
		return type(self).__name__

	def set_internal_input(self,other,idx):
		self.internal_input.extend((idx,other))

	def set_input_idx(self,idx):
		self.input_idx = idx

	def get_input_idx(self):
		return [self.input_index]

	def get_output_idx(self):
		return self.input_idx + self.N - 1

	def get_index_groups(self):
		return [(self.input_idx,self.get_output_idx()+1)]

	def get_input_nodes(self):
		return self.nodes

	def get_output_nodes(self):
		return self.nodes

	def build_nodes(self,input_idx):
		self.set_input_idx(input_idx)
		self.nodes = self.matrix_to_nodes(self.A)

	def matrix_to_nodes(self,A):
		A = A.tocsr()
		nodes = []

		start = self.input_idx
		nodes = [Node(start+idx,start+idx) for idx in range(A.shape[0])]

		for idx,row in enumerate(A):
			node = nodes[idx]
			nonzero = np.nonzero(row.toarray()[0])[0]
			for other_idx in nonzero:
				other = nodes[other_idx]

				weight = A[idx,other_idx]
				node.set_input(other,weight)
				other.set_output(node,weight)

		return nodes

class VAR(Component):
	def __init__(self,M,p):
		p = p + 1 # to agree with common notation
		super(VAR,self).__init__(M*p)
		self.M = M
		self.p = p
		self.build()

	def build(self):
		M = self.M
		p = self.p 
		self.A = [ESN_A("DLR",p,r=1) for i in range(M)]
		self.A = sparse.block_diag(self.A)
		self.A = self.A.tocsr()
		self.B = [ESN_B("SINGLE",M,p,external_input=i) for i in range(M)]
		self.B = np.concatenate(self.B,axis=0)
		self.f = [ESN_f("LIN")]
	
class DIRECT(Component):
	def __init__(self,M):
		super(DIRECT,self).__init__(M)
		self.M = M
		self.build()

	def build(self):
		M = self.M
		self.A = ESN_A("DLR",self.N,r=0)
		self.B = ESN_B("DIRECT",self.N,self.N)
		self.f = [ESN_f("LIN")]

class LEAKY(Component):
	def __init__(self,M,N,r,v):
		super(LEAKY,self).__init__(N)
		self.M = M
		self.r = r
		self.v = v
		self.build()

	def build(self):
		self.A = ESN_A("DIAGONAL",self.N,self.r)
		self.B = ESN_B("SECTIONS",self.M,self.N,v=self.v)
		self.f = [ESN_f("LIN")]

class RODAN(Component):
	def __init__(self,M,N,r=0.5,v=1):
		super(RODAN,self).__init__(N)
		self.M = M
		self.N_init = N
		self.r = r 
		self.v = v
		self.build()

	def build(self):
		M = self.M
		N = self.N_init
		self.A = ESN_A("DLR",self.N,r=self.r)
		self.B = ESN_B("SECTIONS",M,N,v=self.v)
		self.f = [ESN_f("TANH")]

	def get_input_idx(self):
		return list(range(self.input_idx,self.get_output_idx()))

class THRES(Component):
	def __init__(self,M,N,direct_input=True,random_thres=False,turn_on=True):
		super(THRES,self).__init__(2*N)
		self.number = N
		self.M = M
		self.direct_input = direct_input
		self.random_thres = random_thres
		self.turn_on = turn_on
		self.build()

	def build(self):
		M = self.M
		direct_input = self.direct_input
		random_thres = self.random_thres
		turn_on = self.turn_on
		self.A,self.B,self.f = make_thres(M,self.number,direct_input,random_thres,turn_on)

	#def build_nodes(self,input_idx):
	#	self.set_input_idx(input_idx)
	#	self.nodes = [Node(self.input_idx,self.get_output_idx())]

	def get_index_groups(self):
		start = self.input_idx
		N = self.number
		return [(start+0*N,start+1*N),
				(start+1*N,start+2*N)]

	def get_input_nodes(self):
		return self.nodes[:self.number]

	def get_output_nodes(self):
		return self.nodes[-self.number:]

class TRIGGER(Component):
	def __init__(self,M,N,direct_input=True,random_thres=False,turn_on=True):
		super(TRIGGER,self).__init__(3*N)
		self.number = N
		self.M = M
		self.direct_input = direct_input
		self.random_thres = random_thres
		self.turn_on = turn_on
		self.build()

	def build(self):
		M = self.M
		direct_input = self.direct_input
		random_thres = self.random_thres
		turn_on = self.turn_on
		self.A,self.B,self.f = make_trigger(M,self.number,direct_input,random_thres,turn_on)

	#def build_nodes(self,input_idx):
	#	self.set_input_idx(input_idx)
	#	self.nodes = [Node(self.input_idx,self.get_output_idx())]

	def get_index_groups(self):
		start = self.input_idx
		N = self.number
		return [(start+0*N,start+1*N),
				(start+1*N,start+2*N),
				(start+2*N,start+3*N)]

	def get_input_nodes(self):
		return self.nodes[:self.number]

	def get_output_nodes(self):
		return self.nodes[-self.number:]

class HEIGHTSENS(Component):
	def __init__(self,M,N,random_thres):
		super(HEIGHTSENS,self).__init__(N)
		self.M = M
		if random_thres:
			self.r = 1*np.random.random([N,1])
		else:
			self.r = 1
		self.build()

	def build(self):
		self.A = ESN_A("DIAGONAL",self.N,r=1)
		self.B = ESN_B("SECTIONS",self.M,self.N)
		g = ESN_f("INVERSE_DECAY",self.r)
		h = ESN_f("DOUBLE_POSLIN",self.r)
		self.f = [lambda x: g(h(x))]
		#self.f = [ESN_f("TANH")]

# example:
# spec = {"DIRECT": None,"VAR": {"p": 5}, "RODAN": {"N": 200}, "THRES": {"random_thres": True, "N": 20}}
# Will make VAR component with memory 5, direct component (superflous since VAR), one component as described
# in [Rodan2011] with 200 nodes, and 20 threshold components with direct input from externals and random thresholds.

# transfers = [("SOURCE","TARGET",number), ...]
# archs = [(row,column), ...]

class Reservoir:
	def __init__(self,M,spec,mixing_spec,replace):
		self.build(M,spec)
		self.build_nodes()
		self.mix(mixing_spec,replace)

	def get_matrices(self):
		A = self.get_reservoir_matrix()
		B = np.concatenate([comp.B for comp in self.components],axis=0)
		f = flatten([comp.f for comp in self.components])
		idx_groups = self.get_index_groups()

		return A,B,f,idx_groups

	def total_size(self):
		total = 0
		for comp in self.components:
			total += comp.N

		return total

	def build(self,M,spec):
		components = []

		for key,comp_spec in spec:
			#key,comp_spec = spec
			if key == "VAR":
				comp = [VAR(M,**comp_spec)]
			elif key == "DIRECT":
				comp = [DIRECT(M)]
			elif key == "LEAKY":
				comp = [LEAKY(M,**comp_spec)]
			elif key == "RODAN":
				comp = [RODAN(M,**comp_spec)]
			elif key == "THRES":
				comp = [THRES(M,**comp_spec)]
			elif key == "TRIGGER":
				comp = [TRIGGER(M,**comp_spec)]
			elif key == "HEIGHTSENS":
				comp = [HEIGHTSENS(M,**comp_spec)]
			else:
				print("Component not implemented!")
			
			components += comp

		self.components = components

	def mix(self,mixing_spec,replace):
		for transfer in mixing_spec:
			sources = self.get_output_nodes_of_type(transfer[0])
			targets = self.get_input_nodes_of_type(transfer[1])
			print(transfer)
			#print(sources)
			#print(targets)

			if sources != [] and targets != []:
				num = transfer[2]
				selected_sources = np.random.choice(sources,num,replace)
				selected_targets = np.random.choice(targets,num,replace)

				weight = 1
				for source,target in zip(selected_sources,selected_targets):
					source.set_output(target,weight)
					target.set_input(source,weight)

	def get_nodes(self): # return all nodes in reservoir as list
		return flatten([comp.nodes for comp in self.components])

	def get_input_nodes_of_type(self,comp_type): # return all nodes of a certain type
		return flatten([comp.get_input_nodes() for comp in self.components if comp.get_typename() == comp_type])

	def get_output_nodes_of_type(self,comp_type): # return all nodes of a certain type
		return flatten([comp.get_output_nodes() for comp in self.components if comp.get_typename() == comp_type])

	def build_nodes(self):
		lengths = [comp.N for comp in self.components]
		lengths.insert(0,0)
		accum_lengths = np.cumsum(lengths)[:-1]

		for comp,start in zip(self.components,accum_lengths):
			comp.build_nodes(start)

	def get_index_groups(self):
		return flatten([comp.get_index_groups() for comp in self.components])

	def get_reservoir_matrix(self):
		N = self.total_size()
		nodes = self.get_nodes()

		A = sparse.identity(N)*0 # zero matrix
		A = A.tolil()
		print(N)

		for comp in self.components:
			for node in comp.nodes:
				for other,weight in node.inputs:
					A[node.input_idx,other.output_idx] = weight

			start = comp.input_idx
			for i in range(comp.N):
				for j in range(comp.N):
					A[start+i,start+j] = comp.A[i,j]

		A = A.tocsr()
		return A

	def rebuild(self,comp_types,passes_threshold):
		total_rebuilt = 0
		for comp in self.components:
			if comp.get_typename() in comp_types:
				start = comp.input_idx
				end = comp.get_output_idx()+1
				#print(passes_threshold[start:end])
				if not np.any(passes_threshold[start:end]):
					comp.build()
					total_rebuilt += 1

		print("Rebuilt {0:d} components".format(total_rebuilt))
		return self.get_matrices()

	def print_reservoir(self):
		for comp in self.components:
			for node in comp.nodes:
				line = comp.get_typename() + ": "
				line += str(node.input_idx)+" "+str(node.output_idx)+" "
				line += "--> "+" ".join(["({0:d},{1:.2f})".format(other.output_idx,weight) for other,weight in node.inputs])
				line += " | "
				line += " ".join(["({0:d},{1:.2f})".format(other.input_idx,weight) for other,weight in node.outputs]) + " -->"
				print(line)

	def print_significant(self,sig,sig_limit):
		sig_nodes = np.where(sig < sig_limit)[0]
		i = 0
		node_idx = sig_nodes[i]
		#print(sig_nodes)
		#for comp in self.components:
		#	print(comp.get_typename())
		#	print(comp.get_output_idx())
		for comp in self.components:
			#print(comp.get_typename())
			#print(comp.get_output_idx())
		
			if comp.get_output_idx() >= node_idx:
				print("{0:s}: ".format(comp.get_typename()))
			
			sigs = []
			while comp.get_output_idx() >= node_idx:
				#print("{0:s}: {1:.3f}".format(comp.get_typename(),sig[node_idx]))
				sigs.append(sig[node_idx])
				i += 1
				try:
					node_idx = sig_nodes[i]
				except IndexError:
					node_idx = self.total_size()

			if sigs != []:
				print(np.array(sigs))


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

if __name__ == '__main__':	
	r = 0.25
	g = ESN_f("INVERSE_DECAY",r)
	h = ESN_f("DOUBLE_POSLIN",r)
	f = lambda x: g(h(x))

	a = np.linspace(-10,10,1000)

	plt.plot(a,f(a))
	plt.plot(a,a)
	plt.plot(a,np.zeros_like(a))
	plt.show()










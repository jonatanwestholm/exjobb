# models_auxilliary.py
#[Rodan2011] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2011): 131-144.

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

esn_component_sizes = {"VAR":1,"RODAN":1,"DIRECT":1,"THRES":2,"TRIGGER":3}
chain = [0,np.array([0,1]),np.array([0,0,1,1,0]),np.array([0,0,0,1,1,1,0,0,1,0,1]),np.array([0,0,0,0,1,1,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0])]

THRES_HIGH = 100000000
eps = 0.00001
TH = THRES_HIGH

def print_mat(mat):
	print()
	for row in mat:
		print("".join(["{0:.3f}".format(elem).rjust(10,' ') for elem in row]))

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

def rank(lst):
	return [i[0] for i in sorted(enumerate(lst), key=lambda x:x[1])]

def normalize_corr_mat(dep):
	#autoprecisions = forgiving_inv(np.diag(dep))
	autocorr = np.sqrt(np.diag(1/np.diag(dep)))
	#print_mat(autocorr)
	return np.dot(autocorr,np.dot(dep,autocorr))

def shannon_entropy(dists):
	total = sum(flatten(dists))
	entropy = 0
	for dist in dists:
		tot = sum(dist)
		if tot:
			sh = sum([p*np.log2(p/tot) for p in dist if p])
			entropy += sh

	return -entropy/total

def shannon_separators(binary_idxs,X,Y,num):
	print(binary_idxs)
	shannon_gain = []
	#X_pos = X[np.where(Y==1)[0],:]
	#X_neg = X[np.where(Y==0)[0],:]
	#AUB = sum(Y)[0]
	#CUD = X.shape[0]-AUB
	#print(AUB)
	#print(CUD)
	for bin_idx in binary_idxs:
		try:
			A = sum(Y[np.where(X[:,bin_idx]==1)[0]])[0]
			C = sum(X[:,bin_idx])-A
			B = sum(Y[np.where(X[:,bin_idx]==0)[0]])[0]
			D = sum(X[:,bin_idx]==0)-B
		except TypeError:
			A = sum(Y[np.where(X[:,bin_idx]==1)[0]])
			C = sum(X[:,bin_idx])-A
			B = sum(Y[np.where(X[:,bin_idx]==0)[0]])
			D = sum(X[:,bin_idx]==0)-B
		#print(A)
		#A = sum(x_pos)
		#C = AUC-A
		#C = sum(x_neg)
		#D = BUD-D
		#print("A={0:.1f} B={1:.1f} C={2:.1f} D={3:.1f}".format(A,B,C,D))
		#print(shannon_entropy([[A,B],[C,D]]))
		shannon_gain.append(shannon_entropy([[A,C],[B,D]]))

	initial = shannon_entropy([[A+B,C+D]])
	print(initial)
	print(shannon_gain-initial/10000000)
	shannon_rank = rank(shannon_gain)

	plot_variable_splits(X[:,[binary_idxs[idx] for idx in shannon_rank[:num]]],Y,[shannon_gain[idx] for idx in shannon_rank[:num]],2)

	#return binary_idxs[shannon_rank[:num]]
	return [binary_idxs[idx] for idx in shannon_rank[:num]]

def significant_nodes(X,Y,plot=False):
	Y = Y-np.min(Y)
	Y = Y/np.max(Y)
	sigs = []
	seps = []
	for feat in X.T:
		pos = feat[np.where(Y==1)[0]]
		neg = feat[np.where(Y==0)[0]]

		sig,sep = significance_test(pos,neg)

		if plot:
			#plt.scatter(neg,np.zeros_like(neg),color='b')
			#plt.scatter(pos,np.zeros_like(pos),color='r')
			plt.hist([pos,neg],20)
			plt.title("Confidence interval overlap: {0:.3f} \n Split at: {1:.3f}".format(sig,sep))
			plt.xlabel("Value")
			plt.ylabel("Number of time samples")
			plt.show()

		sigs.append(sig)
		seps.append(sep)

	sigs = np.array(sigs)
	seps = np.array(seps)
	#print(sig)
	return sigs,seps

def significance_test(x1,x2):

	mean1 = np.mean(x1)
	mean2 = np.mean(x2)

	std1 = np.std(x1)
	std2 = np.std(x2)

	sep = (mean1*std2+mean2*std1)/(std1+std2)

	quant = 2
	return overlap([mean1-quant*std1,mean1+quant*std1],[mean2-quant*std2,mean2+quant*std2]),sep

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

def fit_svd(X,num,plot=False):	
	__,S,V = np.linalg.svd(X,full_matrices=False)
	
	if plot:
		cumulative_singular_values(S,True)

	return V[:num,:]

def fit_svd_sep(X_pos,X_neg,num,plot=False):
	num = int(num/2)
	Cs_pos = fit_svd(X_pos,num,plot)
	Cs_neg = fit_svd(X_neg,num,plot)

	return np.concatenate([Cs_pos,Cs_neg],axis=0)

def fit_kmeans(X_pos,X_neg,num):
	
	pos = KMeans(n_clusters=num).fit(X_pos)
	neg = KMeans(n_clusters=num).fit(X_neg)

	#pos_centers = pos.cluster_centers_
	#neg_centers = neg.cluster_centers_
	
	#print(pos_centers)
	#print(neg_centers)
	
	#kmeans = KMeans(n_clusters=num).fit(X)
	#print(kmeans.cluster_centers_)

	return pos,neg

def color_ranking(X,Y):
	num = int(X.shape[1]/2)

	Rank = []
	for i,row in enumerate(X):
		row_rank = rank(row) #[i[0] for i in sorted(enumerate(row), key=lambda x:x[1])]
		row_rank = [int(elem < num) for elem in row_rank]
		Rank.append(row_rank)

	Rank = np.array(Rank)
	#print(Rank)

	spacing = 0.5*np.ones_like(Y)

	Rank = np.concatenate([Y,spacing,Rank],axis=1)

	plt.imshow(Rank,interpolation='none',aspect='auto',extent=[-1, X.shape[1], 0, X.shape[0]])
	plt.title("Centroid ranking. Red: positive. Blue: negative.")
	plt.xlabel("Rank")
	plt.ylabel("Time")
	plt.show()

def cumulative_singular_values(S,plot=False):
	S_energy = S#np.cumsum(S)
	S_energy = S_energy#/S_energy[-1]
	if plot:
		plt.plot(S_energy)
		plt.title("Cumulative singular values")
		plt.xlabel("Singular value")
		plt.ylabel("Cumulative relative singular values")
		plt.show()

	return S_energy

def plot_variable_splits(Xs,Y,explanations="",num_bins=50):
	Xs_pos = Xs[np.where(Y==1)[0]]
	Xs_neg = Xs[np.where(Y==0)[0]]

	if explanations == "":
		explanations = list(range(Xs_pos.shape[1]))

	for xs_pos,xs_neg,expl in zip(Xs_pos.T,Xs_neg.T,explanations):
		#print(xs_pos)
		#print(xs_neg)
		plt.figure()
		plt.title(expl)
		plt.hist([xs_pos,xs_neg],num_bins)
		plt.legend(["Positive examples","Negative examples"])
		plt.xlabel("Value")
		plt.ylabel("Number of time samples")

	plt.show()

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
	elif architecture == 'POSNEG_LIN':
		f = lambda x: (x>thres)*x*0.001
	elif architecture == "DOUBLE_POSLIN":
		f = lambda x: (np.abs(x)>thres)*x + (np.abs(x)<=thres)*thres
	elif architecture == "THRES":
		f = lambda x: (x>thres)*1.0
	elif architecture == "INVERSE_DECAY":
		f = lambda x: x - thres**2/x
	elif architecture == "COUNTER":
		f = lambda x: ((np.floor(x/eps)+thres*(np.floor(x/eps)==0)-1)%thres)*eps

	return f

def make_thres(M, N, direct_input, random_thres, turn_on):
	B = ESN_B("SELECTED",M,2*N)
	#B[N:,:] = 0
	if not direct_input:
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
	B[N:2*N,:] = 0
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

def make_expdelay(M, N, order, direct_input):
	size = order*N
	A2A = np.eye(size)
	B2A = np.diag(500*np.ones(size-N),-N)
	for i in range(size):
		B2A[i,i] = -1000
	C2A = np.zeros([size,size])

	A2B = np.eye(size)
	B2B = np.zeros([size,size])
	C2B = -2*TH*np.eye(size)

	A2C = np.zeros([size,size])
	B2C = np.zeros([size,size])
	C2C = np.eye(size)

	A2 = np.concatenate([A2A,A2B,A2C],axis=0)
	B2 = np.concatenate([B2A,B2B,B2C],axis=0)
	C2 = np.concatenate([C2A,C2B,C2C],axis=0)

	A = np.concatenate([A2,B2,C2],axis=1)
	#print_mat(A)
	A = sparse.csr_matrix(A)

	B = B = ESN_B("SELECTED",M,3*size)
	B[N:,:] = 0
	if not direct_input:
		B[:N,:] = 0

	f = [ESN_f("LIN"),ESN_f("POSNEG_LIN",-TH*eps)]+[ESN_f("COUNTER",2**i) for i in range(1,order+1)]

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
	def __init__(self,M,N,r=0.5,v=0.5):
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
		self.B = self.B #*np.random.normal(0,1,[N,1])
		#print(self.B)
		self.f = [ESN_f("TANH")]

	def get_input_idx(self):
		return list(range(self.input_idx,self.get_output_idx()))

class CHAIN(Component):
	def __init__(self,M,order,r=0.5,v=0.5):
		super(CHAIN,self).__init__(M*len(chain[order]))
		self.M = M
		self.order = order
		self.r = r 
		self.v = v
		self.chain_length = len(chain[self.order])
		self.chain = 2*chain[self.order]-1
		self.build()

	def build(self):
		M = self.M
		N = self.N
		self.A = ESN_A("DLR",self.N,r=self.r)
		# Remove connections between feature chains
		A = self.A.tolil()
		for i in range(1,M):
			j = i*self.chain_length
			A[j,j-1] = 0
		self.A = A.tocsr()
		# #
		self.B = ESN_B("SECTIONS",M,N,v=0)
		# Set input patterns for chains
		B = self.B
		for i in range(M):
			start = i*self.chain_length
			end = (i+1)*self.chain_length
			B[start:end,i] = self.chain
		self.B = B
		#print(self.B)
		self.f = [ESN_f("TANH")]

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

class EXPDELAY(Component):
	def __init__(self,M,N,order,direct_input=True):
		super(EXPDELAY,self).__init__(3*order*N)
		self.number = N
		self.M = M
		self.order = order
		self.direct_input = direct_input
		self.build()

	def build(self):
		M = self.M
		order = self.order
		direct_input = self.direct_input
		self.A,self.B,self.f = make_expdelay(M,self.number,self.order,direct_input)

	#def build_nodes(self,input_idx):
	#	self.set_input_idx(input_idx)
	#	self.nodes = [Node(self.input_idx,self.get_output_idx())]

	def get_index_groups(self):
		start = self.input_idx
		N = self.number*self.order
		return [(start+0*N,start+1*N),(start+1*N,start+2*N)]+[(i,i+self.number) for i in range(start+2*N,start+3*N,self.number)]

	def get_input_nodes(self):
		return self.nodes[:self.number]

	def get_output_nodes(self):
		return self.nodes[:self.number*self.order]

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
	def __init__(self,M,spec,mixing_spec):
		self.build(M,spec)
		self.build_nodes()
		self.mix(mixing_spec)

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

		description = []

		for key,comp_spec in spec:
			desc = key+"; {"+", ".join(["{0:s}: {1:s}".format(arg,str(comp_spec[arg])) for arg in comp_spec])+"}"
			print(desc)
			description.append(desc)
			#key,comp_spec = spec
			if key == "VAR":
				comp = [VAR(M,**comp_spec)]
			elif key == "DIRECT":
				comp = [DIRECT(M)]
			elif key == "LEAKY":
				comp = [LEAKY(M,**comp_spec)]
			elif key == "RODAN":
				comp = [RODAN(M,**comp_spec)]
			elif key == "CHAIN":
				comp = [CHAIN(M,**comp_spec)]
			elif key == "THRES":
				comp = [THRES(M,**comp_spec)]
			elif key == "TRIGGER":
				comp = [TRIGGER(M,**comp_spec)]
			elif key == "EXPDELAY":
				comp = [EXPDELAY(M,**comp_spec)]
			elif key == "HEIGHTSENS":
				comp = [HEIGHTSENS(M,**comp_spec)]
			else:
				print("Component not implemented!")
			
			components += comp

		print(", ".join(description))

		self.components = components

	def mix(self,mixing_spec):
		for transfer in mixing_spec:
			sources = self.get_output_nodes_of_type(transfer[0])
			targets = self.get_input_nodes_of_type(transfer[1])
			print(transfer)
			#print(sources)
			#print(targets)

			if sources != [] and targets != []:
				num = transfer[2]
				if len(transfer) > 3:
					replace = transfer[3]
				else:
					replace = False
				selected_sources = np.random.choice(sources,num,replace)
				selected_targets = np.random.choice(targets,num,replace)

				if len(transfer) > 4:
					weight = transfer[4]
				else:
					weight = 0.5
				for source,target in zip(selected_sources,selected_targets):
					sgn = (np.random.random() > 0.5)*2 - 1
					weight *= sgn
					source.set_output(target,weight)
					target.set_input(source,weight)

		print(", ".join(map(str,mixing_spec)))

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

	def get_binary_idxs(self):
		binary_idxs = []
		for comp in self.components:
			if comp.get_typename() == "THRES":
				start = comp.input_idx
				N = comp.number
				binary_idxs.extend(list(range(start,start+N)))
			elif comp.get_typename() == "TRIGGER":
				start = comp.input_idx
				N = comp.number
				binary_idxs.extend(list(range(start+N,start+2*N)))

		return binary_idxs

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

	def print_significant(self,sig,sig_nodes):
		#sig_nodes = np.where(sig <= sig_limit)[0]
		i = 0
		node_idx = sig_nodes[i]
		#print(sig_nodes)
		#for comp in self.components:
		#	print(comp.get_typename())
		#	print(comp.get_output_idx())
		for comp in self.components:
			#print(comp.get_typename())
			#print(comp.get_output_idx())
			idxs = list(range(comp.input_idx,comp.get_output_idx()))
			covered = [node_idx for node_idx in sig_nodes if node_idx in idxs]
		
			if len(covered):
				print("{0:s}: ".format(comp.get_typename()))
			
			sigs = [sig[idx] for idx in covered]
			'''
			while comp.get_output_idx() >= node_idx:
				#print("{0:s}: {1:.3f}".format(comp.get_typename(),sig[node_idx]))
				sigs.append(sig[node_idx])
				i += 1
				try:
					node_idx = sig_nodes[i]
				except IndexError:
					node_idx = self.total_size()
			'''
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
	dists0 = [[10,1]]
	dists1 = [[9,1],[1,0]]

	print(shannon_entropy(dists0))
	print(shannon_entropy(dists1))










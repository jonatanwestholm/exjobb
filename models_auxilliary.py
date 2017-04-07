# models_auxilliary.py
#[Rodan2011] Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2011): 131-144.

import numpy as np
from scipy import sparse

esn_component_sizes = {"VAR":1,"RODAN":1,"DIRECT":1,"THRES":2,"TRIGGER":3}

THRES_HIGH = 1000

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
	elif architecture == "THRES":
		f = lambda x: x>thres

	return f

def make_thres(M, direct_input, random_thres, turn_on):
	if direct_input:
		B = ESN_B("SELECTED",M,2,replace=False)
	else:
		B = ESN_B("SELECTED",M,2)
		B[0,:] = 0

	A = sparse.diags([THRES_HIGH],[-1])

	if random_thres:
		THRES = np.random.normal(0,1)
	else:
		THRES = 0

	if turn_on:
		f = [ESN_f("THRES",THRES),ESN_f("POSLIN",THRES_HIGH)]
	else:
		A = -A
		f = [ESN_f("THRES",THRES),ESN_f("POSLIN",0)]

	return A,B,f

def make_trigger(M, direct_input, random_thres, turn_on):
	if direct_input:
		B = ESN_B("SELECTED",M,3)
		B[1,:] = 0
	else:
		B = ESN_B("SELECTED",M,3)
		B[1:,:] = 0

	if random_thres:
		THRES = np.random.normal(0,1)
	else:
		THRES = 0

	if turn_on:
		f = [ESN_f("THRES",THRES), ESN_f("THRES",0), ESN_f("POSLIN",THRES_HIGH)]
		A = sparse.diags([[0,THRES_HIGH,0],[1,THRES_HIGH]],[0,-1])
	else:
		f = [ESN_f("THRES",THRES), ESN_f("THRES",0), ESN_f("POSLIN",0)]
		A = sparse.diags([[0,THRES_HIGH,0],[1,-THRES_HIGH]],[0,-1])

	return A,B,f

class Component:
	def __init__(self,N): # f_arch a.k.a. activation
		self.N = N
		self.internal_input = []

	def get_matrices(self):
		return [self.A],[self.B],self.f

	def get_typename(self):
		return type(self).__name__

	def set_internal_input(self,other):
		self.internal_input.extend(other)

	def set_input_idx(self,idx):
		self.input_idx = idx

	#def get_input_idx(self):
	#	return self.input_index

	def get_output_idx(self):
		return self.input_idx + self.N - 1

	def get_index_groups(self):
		if self.common_f:
			return [(self.input_idx,self.get_output_idx()+1)]
		else:
			return [(i,i+1) for i in range(self.input_idx,self.get_output_idx()+1)]

class VAR(Component):
	def __init__(self,M,p):
		p = p + 1 # to agree with common notation
		super(VAR,self).__init__(M*p)
		self.common_f = True

		self.A = [ESN_A("DLR",p,r=1) for i in range(M)]
		self.A = sparse.block_diag(self.A)
		self.B = [ESN_B("SINGLE",M,p,external_input=i) for i in range(M)]
		self.B = np.concatenate(self.B,axis=0)
		self.f = [ESN_f("LIN")]
	
class DIRECT(Component):
	def __init__(self,M):
		super(DIRECT,self).__init__(M)
		self.common_f = True

		self.A = ESN_A("DLR",self.N,r=0)
		self.B = ESN_B("DIRECT",self.N)
		self.f = [ESN_f("LIN")]

class RODAN(Component):
	def __init__(self,M,N):
		super(RODAN,self).__init__(N)
		self.common_f = True

		self.A = ESN_A("DLR",self.N,r=0.5)
		self.B = ESN_B("SECTIONS",M,N)
		self.f = [ESN_f("TANH")]

class THRES(Component):
	def __init__(self,M,N=1,direct_input=True,random_thres=False,turn_on=True):
		super(THRES,self).__init__(2)
		self.common_f = False

		self.A,self.B,self.f = make_thres(M,direct_input,random_thres,turn_on)

class TRIGGER(Component):
	def __init__(self,M,N=1,direct_input=True,random_thres=False,turn_on=True):
		super(TRIGGER,self).__init__(3)
		self.common_f = False

		self.A,self.B,self.f = make_trigger(M,direct_input,random_thres,turn_on)

'''
components = {"AR": ["FLEX","DLR",1,"SECTIONS_INIT",1,"LIN"],
			  "RODAN": ["FLEX","DLR",0.5,"SECTIONS",1,"TANH"],
			  "DIRECT": ["FLEX","DLR",0,"DIRECT",1,"LIN"],
			  "THRES": ["THRES"],
			  "TRIGGER": ["TRIGGER"],
			  "HEIGHTSENS": ["HEIGHTSENS"],
			  "ANOM": ["ANOM"]}
'''

# example:
# spec = {"DIRECT": None,"VAR": {"p": 5}, "RODAN": {"N": 200}, "THRES": {"random_thres": True, "N": 20}}
# Will make VAR component with memory 5, direct component (superflous since VAR), one component as described
# in [Rodan2011] with 200 nodes, and 20 threshold components with direct input from externals and random thresholds.

def compound_ESN(spec,M):
	components = []

	for key in spec:
		comp_spec = spec[key]
		if key == "VAR":
			comp = [VAR(M,**comp_spec)]
		elif key == "DIRECT":
			comp = [DIRECT(M)]
		elif key == "RODAN":
			comp = [RODAN(M,**comp_spec)]
		elif key == "THRES":
			comp = [THRES(M,**comp_spec) for i in range(comp_spec["N"])]
		elif key == "TRIGGER":
			comp = [TRIGGER(M,**comp_spec) for i in range(comp_spec["N"])]
		
		components += comp

	return components

# turns list of lists into list
def flatten(lst):
	return [elem for sublist in lst for elem in sublist]

def set_input_idxs(components):
	lengths = [comp.N for comp in components]
	lengths.insert(0,0)
	accum_lengths = np.cumsum(lengths)[:-1]

	for comp,start in zip(components,accum_lengths):
		comp.set_input_idx(start)

def get_index_groups(components):
	set_input_idxs(components)
	return flatten([comp.get_index_groups() for comp in components])

def generate_matrices(components):
	A = sparse.block_diag([comp.A for comp in components])
	B = np.concatenate([comp.B for comp in components],axis=0)
	f = flatten([comp.f for comp in components])

	idx_groups = get_index_groups(components)

	A = A.tocsr()
	for comp in components:
		i = comp.input_idx
		for j in comp.internal_input:
			A[i,j] = 1 # direct for now


	return A,B,f,idx_groups









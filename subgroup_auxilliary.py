# subgroup_auxilliary.py

# The methods in this module should have very well-defined tasks
# They should not call any methods in the main subgroup module
# They should not take a diffuse parameter called "args", all parameters should have names

import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pp
import models as Models

## General

class Subgroup_collection:
	# N is number of features in dataset
	def __init__(self,N,models):
		self.N = N
		self.models = []
		self.candidates = []
		self.remaining = []
		self.add(models,"MODELS")

	def get_covered(self):
		established = flatten([mod.subgroup for mod in self.models])
		candidate = flatten([mod.subgroup for mod in self.candidates])
		return established + candidate

	def get_remaining(self):
		covered = self.get_covered()
		return [i for i in range(self.N) if i not in covered]

	def add(self,mods,status):
		covered = self.get_covered()
		for mod in mods:
			for feature in mod.subgroup:
				if feature in covered:
					print("Feature already covered by collection!!")
					raise
				else:
					covered.append(feature)
			if status == "MODELS":
				self.models.append(mod)
			elif status == "CANDIDATES":
				self.candidates.append(mod)
			elif status == "REMAINING":
				self.remaining.append(mod)

	def reset(self):
		self.candidates = []
		self.remaining = []

# turns list of lists into list
def flatten(lst):
	return [elem for sublist in lst for elem in sublist]

def is_tensor(X,order):
	#print(X)
	#print(np.shape(X))
	if sum(np.array(np.shape(X)) > 1) == order:
		return True
	else:
		return False

# it is decided that this should always work
def num_features(dat):
	if is_tensor(dat,1):
		return 1
	else:
		return np.shape(dat)[1]

## Candidate generation

def map_idx(arr,idx):
	return [arr[i] for i in idx]

def print_mat(mat):
	print()
	for row in mat:
		print("".join(["{0:.3f}".format(elem).rjust(10,' ') for elem in row]))

'''
def forgiving_inv(arr):
	inverse = 1/arr
	return inverse
	
	for i,inv in enumerate(inverse):
		if not np.isfinite(inv):
			inverse[i] = 0
	return inverse
	
	#return np.array([1/elem for elem in arr if elem else 0])
'''

def normalize_corr_mat(dep):
	if is_tensor(dep,2):
		#autoprecisions = forgiving_inv(np.diag(dep))
		autocorr = np.sqrt(np.diag(1/np.diag(dep)))
		#print_mat(autocorr)
		return np.dot(autocorr,np.dot(dep,autocorr))
	elif is_tensor(dep,3):
		autocorr = np.sqrt(np.diag(1/np.diag(dep[0,:,:])))
		for i in range(np.shape(dep)[0]):
			dep[i,:,:] = np.dot(autocorr,np.dot(dep[i,:,:],autocorr))
		return dep
		
def linear_dependence(dat,lag):
	N = num_features(dat)
	dep = np.dot(dat.T,dat)
	#print_mat(dep)
	for i in range(1,lag):
		dep += np.dot(dat[i:,:].T,dat[:-i,:])

	#dep = normalize_corr_mat(dep)
	#print_mat(dep)
	return dep

def autocorr(dat,max_ord):
	N = num_features(dat)
	T = np.shape(dat)[0]

	y = np.zeros([max_ord,N,N])
	y[0,:,:] = np.dot(dat.T,dat)
	for i in range(1,max_ord):
		y[i,:,:] = (np.dot(dat[i:,:].T,dat[:-i,:]) + np.dot(dat[:-i,:].T,dat[i:,:]))/2
	
	return y/T

def whiteness_test(data,explanations,max_ord=40):
	N = num_features(data[0])
	
	acorr = sum([autocorr(dat,max_ord) for dat in data])
	acorr = normalize_corr_mat(acorr)
	T = sum([np.shape(dat)[0] for dat in data])
	for i in range(N):
		for j in range(i):
			plt.figure()
			plt.title("Cross-ACF for {0:s} \n and {1:s}".format(explanations[i],explanations[j]))
			plt.xlabel("order")
			plt.plot(acorr[:,i,j])
			plt.plot([0,max_ord],[2/np.sqrt(T)]*2,'r')
			plt.plot([0,max_ord],[-2/np.sqrt(T)]*2,'r')
			plt.legend(["acf","whiteness"])
	plt.show()

def proportional_random_select(arr):
	c_arr = np.cumsum(arr)
	#print(c_arr)
	c_arr = c_arr/c_arr[-1]
	
	u = np.random.random()
	i = 0
	while u > c_arr[i]:
		i += 1

	return i

def greedy_random_cand(dep,num):
	dep = np.abs(dep)
	for i in range(len(dep)):
		dep[i,i] = 0

	idx0 = proportional_random_select(np.sum(dep,axis=0))
	#print(idx0)
	idxs = [idx0]
	for i in range(1,num):
		#print_mat(dep)
		#print()
		dep_sum = np.sum(dep[idxs,:],axis=0)
		selected = proportional_random_select(dep_sum)
		#print(selected)
		for idx in idxs:
			dep[selected,idx] = 0
			dep[idx,selected] = 0

		idxs.append(selected)

	#print_mat(dep)
	#print()

	return idxs

## Training

def train_varma(data,subgroup,p,q,re_series,rw_series):
	N = num_features(data[0])
	print(N)
	#p = args.settings["VARMA_p"]
	#q = args.settings["VARMA_q"]

	A_hist = np.zeros([1,N,N*p])
	C_hist = np.zeros([1,N,N*q])

	orders = [N,p,q]
	mod = Models.VARMA(orders)
	i = 0
	for dat in data:
		if is_tensor(dat,1):
			dat = np.reshape(dat,[len(dat),1])

		mod.reset()
		for y in dat[:p,:]:
			mod.update(y)
		A_h,C_h = mod.annealing(dat[p:,:],re_series,rw_series,initiate= i == 0) #initiate = i==0
		if i >= 0:
			A_hist = np.concatenate([A_hist,A_h[-5:]],axis=0)
			C_hist = np.concatenate([C_hist,C_h[-5:]],axis=0)
		i += 1
	mod.reset()

	A_hist = A_hist[1:]
	C_hist = C_hist[1:]

	#plot_train(A_hist,C_hist,train_type="VARMA")
	A = np.mean(A_hist,axis=0)
	C = np.mean(C_hist,axis=0)
	#print_mat(mod.A)
	#print_mat(mod.C)
	print_mat(A)
	print_mat(C)


	#gt_A = -np.array([[0.1,0.5001,0.2],[-0.2,0.1,0.5001],[0.5001,0.2,-0.1]])
	#print("Incremental, error: " + str(np.linalg.norm(gt_A-mod.A[:3,:3])))
	#print("Averaging, error: " + str(np.linalg.norm(gt_A-A[:3,:3])))
	
	mod.set_A_C(A,C)

	mod.subgroup = subgroup

	return mod

def plot_train(A_hist,C_hist,train_type):
	if A_hist != []:
		N,M = A_hist[0,:,:].shape
		for j in range(N):
			plt.figure()
			plt.title("AR coefficients for y_{0:d}".format(j+1))
			for i in range(M):
				plt.plot(A_hist[:,j,i])
			#for i in range(k*p):
			#	plt.plot([0,N],[-A[j][i]]*2)
			if train_type == "VARMA":
				p = int(M/N)
				legends = flatten([["a({0:d},{1:d})".format(jx+1,ix+1) for jx in range(N)] for ix in range(p)]) # + flatten([["a({0:d},{1:d})_gt".format(jx+1,ix+1) for ix in range(p)] for jx in range(k)])
				#print([["a({0:d},{1:d})_gt".format(jx+1,ix+1) for ix in range(p)] for jx in range(k)])
				plt.legend(legends) #["a{0:d}".format(i+1) for i in range(k*p)]+["a{0:d}_gt".format(i+1) for i in range(p)])
	if C_hist != []:
		N,M = C_hist[0,:,:].shape
		for j in range(N):
			plt.figure()
			plt.title("MA coefficients for y_{0:d}".format(j+1))
			for i in range(M):
				plt.plot(C_hist[:,j,i])
		#for i in range(q):
		#	plt.plot([0,N],[C[0][i+1]]*2)
		if train_type == "VARMA":
			q = int(M/N)
			plt.legend(["c{0:d}".format(i+1) for i in range(q)]) #+["c{0:d}_gt".format(i+1) for i in range(q)])

	plt.show()

def train_esn(mod,data,orders,burn_in,tikho):

	__,size_out,size_label = orders
	C_hist = np.zeros([1,size_label,size_out])

	#if batch_train:
	for dat in data:
		mod.reset()
		for y in dat[:burn_in,:]:
			mod.update(y)
		C_h = mod.learn_batch(dat[burn_in:,:],tikho=tikho)
		#print(C_h.shape)
		#print(C_hist.shape)
		C_hist = np.concatenate([C_hist,C_h],axis=0)
		
	'''			
	else:
		i = 0
		for dat in data:
			if is_tensor(dat,1):
				dat = np.reshape(dat,[len(dat),1])

			mod.reset()
			for y in dat[:burn_in,:]:
				mod.update(y)
			C_h = mod.annealing(dat[burn_in:,:],re_series,rw_series,initiate= i == 0) #initiate = i==0
			if i >= 0:
				#print(C_hist.shape)
				#print(C_h.shape)
				C_hist = np.concatenate([C_hist,C_h[:-1]],axis=0)
			i += 1
	'''

	mod.reset()

	C_hist = C_hist[1:]

	#plot_train([],C_hist,train_type="ESN")
	C = np.mean(C_hist,axis=0)
	#print_mat(mod.A)
	#print_mat(mod.C)
	print_mat(C)

	#gt_A = -np.array([[0.1,0.5001,0.2],[-0.2,0.1,0.5001],[0.5001,0.2,-0.1]])
	#print("Incremental, error: " + str(np.linalg.norm(gt_A-mod.A[:3,:3])))
	#print("Averaging, error: " + str(np.linalg.norm(gt_A-A[:3,:3])))
	
	mod.set_Cw(C)

	return mod


def impending_failure(data,names,dataset,failure_horizon,style):
	if dataset == "TURBOFAN":
		for dat in data:
			X,y = impending_failure_datapoints(dat,True,failure_horizon,style)
			yield X,y
	elif dataset == "BACKBLAZE":
		for dat,name in zip(data,names):
			failure = "_fail" in name
			X,y = impending_failure_datapoints(dat,failure,failure_horizon,style)
			yield X,y

def impending_failure_datapoints(dat,failure,failure_horizon,style):
	N,M = dat.shape
	if failure:
		X = dat
		if style == "SVC":
			y = np.concatenate([-np.ones([N-failure_horizon,1]), np.ones([failure_horizon,1])])
		elif style == "MLP":
			multiplier = 3 #int(N/failure_horizon)

			neg_X = dat[:-failure_horizon,:]
			pos_X = dat[-failure_horizon:,:]
			X = np.concatenate([neg_X]+multiplier*[pos_X])
			y = np.concatenate([-np.ones([N-failure_horizon,1])] + multiplier*[np.ones([failure_horizon,1])])
		elif style == "SVR":
			far_to_fail = failure_horizon*np.ones([N-failure_horizon,1])
			close_to_fail = -np.array(range(-failure_horizon+1,1),ndmin=2).T
			y = np.concatenate([far_to_fail,close_to_fail])
	else:
		X = dat[:-failure_horizon,:]
		if style in ["SVC","MLP"]:
			y = -np.ones([N-failure_horizon,1])
		elif style == "SVR":
			y = failure_horizon*np.ones([N-failure_horizon,1])

	return X,y

'''
def remaining_features(N,models):
	remains = list(range(N))
	for mod in models:
		for feature in mod.subgroup:
			remains.remove(feature)
	return remains
'''

## Testing

def reset_models(models):
	for mod in models:
		mod.reset()

def update_models(data,models):
	for mod in models:
		mod.update_array(data[:,mod.subgroup])

def predict_data(dat,models,k):
	pred_mat = np.zeros_like(dat)
	i = 0
	#X_data = np.zeros([models[0].N,dat.shape[0]])
	for sample in dat:
		pred_array = np.zeros_like(sample)

		for mod in models:
			mod.update(sample[mod.subgroup])
			pred_array[mod.subgroup] = mod.predict(k)

		#X_data[:,i] = np.ravel(models[0].X)

		pred_mat[i,:] = pred_array
		i += 1

	#plt.plot(X_data.T)
	#plt.show()

	return pred_mat

# Visualization

def classification_plot(pred,gt,style,failure_horizon):
	for pred_arr,gt_arr in zip(pred,gt):
		if style in ["SVC","MLP"]:
			pred_arr = (pred_arr+1)/2
			gt_arr = (gt_arr+1)/2
		elif style == "SVR":
			pass

		length = 5
		pred_arr = pp.filter([pred_arr],np.array([1]*length)/length,np.array([1]))[0]
		#print(pred_arr)

		plt.figure()
		plt.plot(pred_arr,'b')
		plt.plot(gt_arr,'r')
		if style in ["SVC","MLP"]:
			plt.axis([0,len(pred_arr), -0.1, 1.1])
			plt.legend(["Predicted","Ground Truth"],loc=2)
		elif style == "SVR":
			plt.axis([length,len(pred_arr), 0, failure_horizon+1])
			plt.legend(["Predicted","Ground Truth"],loc=3)
	plt.show()

# The methods in this module should have very well-defined tasks
# They should not call any methods in the main subgroup module
# They should not take a diffuse parameter called "args", all parameters should have names

import numpy as np
import matplotlib.pyplot as plt
import copy
import preprocessing as pp
#import models as Models
import scipy.signal as ssignal

## General

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

def normalize_01(arr):
	arr -= np.min(arr)
	arr /= np.max(arr)
	return arr

## Candidate generation

def map_idx(arr,idx):
	return [arr[i] for i in idx]

def print_mat(mat):
	print()
	for row in mat:
		print("".join(["{0:.3f}".format(elem).rjust(10,' ') for elem in row]))

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

def classification_plot(test_data,pred,gt,names,dataset,model):
	for test_dat,pred_arr,gt_arr,name in zip(test_data,pred,gt,names):
		length = 1
		pred_arr = pp.filter([pred_arr],np.array([1]*length)/length,np.array([1]))[0]
		#print(pred_arr)

		if 0: #dataset in ["BACKBLAZE"]:
			x = sorted(list(range(-len(pred_arr),0)))
		else:
			x = list(range(len(pred_arr)))

		plt.figure()
		if dataset in ["DODGERS"]:
			plt.plot(x,test_dat[:,0],'g')

		plt.plot(x,pred_arr,'b')
		plt.plot(x,gt_arr,'r')
		plt.title("Dataset: {0:s}. Unit: {1:s}. Model: {2:s}".format(dataset,name,model))
		plt.title("Response: {0:s}".format(name))
		if 0: #dataset in ["BACKBLAZE"]:
			plt.axis([-len(pred_arr),0, min(gt_arr)-0.1,max(gt_arr)+0.1])
		else:
			plt.axis([0,len(pred_arr), min(gt_arr)-0.1,max(gt_arr)+0.1])
		if dataset in ["DODGERS"]:
			plt.legend(["Input","Predicted","Ground Truth"])
		else:
			plt.legend(["Response","Input"],loc='best')
		#plt.xlabel("Days before failure")
		plt.xlabel("Sample no. (time)")
		plt.ylabel("Value")
	plt.show()

def classification_stats(GG,PG,PP):
	spec = PG/GG
	prec = PG/PP
	#am = (spec+prec)/2
	hm = 2/(1/spec + 1/prec)

	#spec = spec[0][0]
	#prec = prec[0][0]
	#am = am[0][0]
	#hm = hm[0][0]
	#print(spec)
	#print(prec)
	#print(am)
	#print(hm)

	return spec,prec,hm

def nz_intervals(G):
	G = copy.copy(G)
	G[0] = 0
	G = np.concatenate([G,[[0]]],axis=0)
	G = 1.0*(G != 0)
	G = np.diff(G,axis=0)
	return [(start,end) for start,end in zip(np.where(G==1)[0],np.where(G==-1)[0])]

def interval_hits(P,G):
	intervals = nz_intervals(G)
	if not len(intervals):
		return np.array([0.0])
	total = 0
	for interval in intervals:
		start,end = interval
		#print(P[start:end])
		total += sum(P[start:end]) != 0

	return total/len(intervals)

def burn_down_graph(pred,thres):
	thres = [0]+thres
	N = len(pred)
	for th in thres:
		warning_times = []
		for P in pred:
			if th:
				C = np.array([1]*th)/th
				P = ssignal.lfilter(C,1,P.T).T
				try:
					warning_times.append(np.min(np.where(P==1)[0])-len(P))
				except ValueError:
					warning_times.append(-1)
			else:
				warning_times.append(-len(P))

		warning_times = sorted(warning_times)
		print(warning_times)

		remaining = np.ones([-warning_times[0]+1,1])
		time_old = 0
		for i,time in enumerate(warning_times):
			remaining[time_old:time] = i/N
			time_old = time

		remaining = remaining[:-1]
		x = np.array(list(range(warning_times[0],0)))

		#print(remaining.shape)
		#print(x.shape)

		plt.plot(x,remaining)
	plt.xlabel("Days before failure")
	plt.ylabel("Share of units that have had warning")
	plt.title("Warning accuracy")
	plt.legend(["Data availability"]+["{0:d} in a row".format(th) for th in thres[1:]],loc=2)
	plt.show()








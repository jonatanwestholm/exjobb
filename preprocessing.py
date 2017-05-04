# preprocessing.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import time
import scipy.signal as ssignal

## I/O

def read_file(name,elemsep,linesep,readlines,mapping=None):
	f = open(name,'r').read()

	if mapping == None:
		data = [read_line(line,elemsep,greedy_numeric) for line in f.split(linesep)]
	else:
		data = [read_line(line,elemsep,mapping) for line in f.split(linesep)]

	if readlines == "all":
		return data
	else:
		return data[:int(readlines)]

def read_line(line,elemsep,mapping):
	if not line:
		return []
	try:
		line_data = list(map(mapping, line.strip().split(elemsep)))
		#print(line_data)
		return line_data
	except ValueError:
		return []

def greedy_numeric(string):
	if not string:
		return np.nan
	try:
		return float(string)
	except ValueError:
		return np.nan

def write_line(line,elemsep):
	return elemsep.join([str(elem) for elem in line])

def write_data(dat,linesep,elemsep):
	return linesep.join([write_line(line,elemsep) for line in dat])

def filter_wrt(data,key,value):
	return data[data[:,key]==value,:]

def filter_wrt_function(data,condition):
	return [dat for dat in data if condition(dat)]

def just_the_names(filenames):
	return [filename.split('/')[-1] for filename in filenames]

## Mathematical operations

def differentiate(data):
	return [np.diff(dat,axis=0) for dat in data]

def smooth(data,length):
	C = np.array([1]*length)/length
	A = 1
	return [ssignal.lfilter(C,A,dat.T).T for dat in data]

def filter(data,C,A):
	return [ssignal.lfilter(C,A,dat.T).T for dat in data]

def sin_signal(num,period):
	signal = np.linspace(0,num,num+1)[:-1]
	signal = signal*2*np.pi/period
	signal = np.sin(signal)
	signal = signal.reshape([num,1])
	return signal

def cos_signal(num,period):
	signal = np.linspace(0,num,num+1)[:-1]
	signal = signal*2*np.pi/period
	signal = np.cos(signal)
	signal = signal.reshape([num,1])
	return signal

def remove_harmonic_trend(dat,T):
	N = len(dat)
	X = np.ones([N,3])
	X[:,1] = np.ravel(sin_signal(N,T))
	X[:,2] = np.ravel(cos_signal(N,T))

	theta = np.linalg.lstsq(X,dat)[0]

	return dat - np.dot(X,theta)

def normalize(dat,return_mean_max=False,leave_zero=False):
	dat_mean = [np.mean(feat) for feat in dat.T]
	dat = np.array([feat-feat_mean for feat,feat_mean in zip(dat.T,dat_mean)]).T
	#dat_max = [np.max(np.abs(feat)) for feat in dat.T]
	dat_max = [np.std(feat) for feat in dat.T]

	if leave_zero:
		for i,feat_max in enumerate(dat_max):
			if feat_max < 10**-8:
				dat_max[i] = 1

	dat = np.array([feat/feat_max for feat,feat_max in zip(dat.T,dat_max)]).T
	#dat = np.array([(feat-0)/feat_max for feat,feat_max in zip(dat.T,dat_max)]).T

	if return_mean_max:
		return dat,dat_mean,dat_max
	else:
		return dat

def normalize_ref(dat,dat_mean,dat_max):
	return np.array([(feat - feat_mean)/feat_max for feat,feat_mean,feat_max in zip(dat.T,dat_mean,dat_max)]).T	
	#return np.array([(feat - 0)/feat_max for feat,feat_mean,feat_max in zip(dat.T,dat_mean,dat_max)]).T	

def normalize_all(data,leave_zero=False):
	__,dat_mean,dat_max = normalize(np.concatenate(data,axis=0),return_mean_max=True,leave_zero=leave_zero)
	#print(dat_mean)
	#print(dat_max)
	return [normalize_ref(dat,dat_mean,dat_max) for dat in data]

#def only_numeric(data):	
#	first_row = data[0][0,:]
#	is_numeric = [elem for elem in first_row if not np.isfinite(elem)]
#	only_num = [dat[:,is_numeric] for dat in data]
#	return only_num,is_numeric

def count_numeric(arr):
	return np.sum(np.isfinite(arr))

def numeric_idxs(data):	
	first_row = data[0][0,:]
	is_numeric = set([i for i,elem in enumerate(first_row) if np.isfinite(elem)])
	for dat in data[1:]:
		#print(is_numeric)
		first_row = dat[0,:]
		#print(first_row)
		is_numeric = set.intersection(is_numeric,set([i for i,elem in enumerate(first_row) if np.isfinite(elem)]))
	#only_num = [dat[:,is_numeric] for dat in data]
	return is_numeric

def changing_idxs(data):
	dat0 = np.concatenate(data,axis=0)
	#for row in dat0:
	#	#print(len(np.where(np.isfinite(feat))[0]))
	#	if len(np.where(np.isfinite(row))[0]) < 10:
	#		print(row)
	dat_std = [np.std(feat) for feat in dat0.T]
	#print(dat_std)
	changing = [i for i,item in enumerate(dat_std) if item > 10**-8]
	#print(changing)

	return changing

## Managing data

def remove_small_samples(data,limit):
	return [dat for dat in data if np.shape(dat)[0] > limit]

def has_missing(dat):
	return np.any(np.isnan(dat))

def remove_instances_with_missing(data):
	no_missing = [i for i in range(len(data)) if not has_missing(data[i])]
	data = [data[i] for i in no_missing]
	return data,no_missing

def remove_unchanging(data):
	changing = changing_idxs(data)

	data = [dat[:,changing] for dat in data]

	return data,changing

# Generating ground truth

def impending_failure(data,failed,failure_horizon,test_type):
	#if dataset in ["TURBOFAN","ESN_SIM"]:
	#	for dat in data:
	#		X,y = impending_failure_datapoints(dat,True,failure_horizon,test_type)
	#		yield X,y
	#elif dataset == "BACKBLAZE":
	for dat,failure in zip(data,failed):
		#failure = "_fail" in name
		X,y = impending_failure_datapoints(dat,failure,failure_horizon,test_type)
		yield X,y

def impending_failure_datapoints(dat,failure,failure_horizon,test_type):
	N,M = dat.shape
	if failure:
		X = dat
		if test_type == "CLASSIFICATION":
			y = np.concatenate([np.zeros([N-failure_horizon,1]), np.ones([failure_horizon,1])])
		elif test_type == "REGRESSION":
			far_to_fail = failure_horizon*np.ones([N-failure_horizon,1])
			close_to_fail = -np.array(range(-failure_horizon+1,1),ndmin=2).T
			y = np.concatenate([far_to_fail,close_to_fail])
	else:
		X = dat[:-failure_horizon,:]
		if test_type == "CLASSIFICATION":
			y = np.zeros([N-failure_horizon,1])
		elif test_type == "REGRESSION":
			y = failure_horizon*np.ones([N-failure_horizon,1])

	return X,y

# split data into train and test
def split(data,gt,split_method,train_share=0.6,test_share=0.2,names="",return_names=False):
	split_method = split_method
	if split_method == "TIMEWISE":	
		#train_share = 0.6
		#test_share = 0.2
		N = [np.shape(dat)[0] for dat in data]
		Split1 = [int(np.floor(train_share*n)) for n in N]
		Split2 = [int(np.floor((train_share+test_share)*n)) for n in N] 

		train_data = [dat[0:split1,:] for dat,split1 in zip(data,Split1)]
		train_gt = [gt_inst[0:split1,:] for gt_inst,split1 in zip(gt,Split1)]
		test_data = [dat[split1:split2,:] for dat,split1,split2 in zip(data,Split1,Split2)]
		test_gt = [gt_inst[split1:split2,:] for gt_inst,split1,split2 in zip(gt,Split1,Split2)]

		train_names = names
		test_names = names
		
	elif split_method == "UNITWISE":
		#train_share = 0.2
		#test_share = 0.2
		N = len(data)
		split1 = int(np.floor(train_share*N))
		split2 = int(np.floor((train_share+test_share)*N))
		print(split1)
		print(split2)
		print(names)

		train_data = data[0:split1]
		train_gt = gt[0:split1]
		test_data = data[split1:split2]
		test_gt = gt[split1:split2]

		train_names = names[0:split1]
		test_names = names[split1:split2]

	if return_names:
		return train_data,train_gt,test_data,test_gt,train_names,test_names
	else:
		return train_data,train_gt,test_data,test_gt

## Visualizations

def display_parallel(data,explanations):
	j = 1
	for dat in data:
		dat = normalize(dat)
		expl = []
		plt.figure()
		for i in range(np.shape(dat)[1]):
			feat = dat[:,i]
			if not np.isnan(feat[0]):
				expl.append(explanations[i])
				plt.plot(feat)

		plt.legend(expl)
		plt.title("Sample nbr {0:d}".format(j))
		j += 1

	plt.show()

explorable_types = ['list','dict','ndarray','void']
def explore(data,branch_limit,expansion_limit,head=""):
	#print(type(data))
	#print(len(data))
	try:
		print(head + "{0:s}_{1:d}".format(type(data).__name__,len(data)))
		if len(data) < branch_limit and type(data).__name__ in explorable_types: #[list,dict,type(np.ndarray([])),type(np.void())]:
			if isinstance(data,dict):
				i = 0
				for elem in data:
					if i < expansion_limit:
						explore(data[elem],branch_limit,expansion_limit,head+"{0:s}  ".format(elem))
					else:
						break
					i += 1
			else:
				i = 0
				for elem in data:
					if i < expansion_limit:
						explore(elem,branch_limit,expansion_limit,head+"{0:d}  ".format(i))
					else:
						break
					i += 1	
	except TypeError:
		print(head + "{0:s}".format(type(data).__name__))

def main(args):
	filename = args.filename
	datatype = args.datatype

	if datatype == "SEQUENTIAL":
		data = read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines)
		data.remove([])
		data = np.array(data)

		#data = filter_wrt(data,0,2)

		while True:
			x = input('Which feature do you want to look at? ')
			x = int(x)
		
			plt.plot(data[:,x])
			plt.show()
		
	elif datatype == "INSTANCE":
		pattern = '[0-9-]*.csv'
		#pattern = args.pattern
		#serial_number = "MJ0351YNG9Z0XA"
		location = 1

		melt_instance(args,filename,pattern,location)

	elif datatype == "LAYERED":
		data = scipy.io.loadmat(filename)
		explore(data,11000,25)
			

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',dest = 'filename',default="",help='Name of input file')
    parser.add_argument('-t','--target',dest = 'target',default="",help='Name of target directory for melting')
    parser.add_argument('-p','--pattern',dest = 'pattern',default="",help='Input file pattern (regex)')    
    parser.add_argument('-d','--datatype',dest = 'datatype',default="SEUQENTIAL",help='Type of data (e.g. SEQUENTIAL,INSTANCE,LAYERED)')
    parser.add_argument('-c','--readlines',dest = 'readlines',default="all",help='Number of lines to read')
    parser.add_argument('-e','--elemsep',dest = 'elemsep',default='\t',help='Element Separator')
    parser.add_argument('-l','--linesep',dest = 'linesep',default='\n',help='Line Separator')
    args = parser.parse_args()
    main(args)
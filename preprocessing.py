# preprocessing.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import time

def read_file(name,elemsep,linesep,readlines):
	f = open(name,'r').read()

	data = [read_line(line,elemsep,greedy_numeric) for line in f.split(linesep)]

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

def only_numeric(data):	
	first_row = data[0][0,:]
	is_numeric = np.array([not np.isnan(elem) for elem in first_row],dtype=bool)
	only_num = [dat[:,is_numeric] for dat in data]
	return only_num

def write_line(line,elemsep):
	return elemsep.join([str(elem) for elem in line])

def write_data(dat,linesep,elemsep):
	return linesep.join([write_line(line,elemsep) for line in dat])

def filter_wrt(data,key,value):
	return data[data[:,key]==value,:]

def filter_wrt_function(data,condition):
	return [dat for dat in data if condition(dat)]

def normalize(dat,return_mean_std=False):
	dat_mean = [np.mean(feat) for feat in dat.T]
	dat_std = [np.std(feat) for feat in dat.T]

	dat = np.array([(feat - feat_mean)/feat_std for feat,feat_mean,feat_std in zip(dat.T,dat_mean,dat_std)]).T

	if return_mean_std:
		return dat,dat_mean,dat_std
	else:
		return dat

def normalize_ref(dat,dat_mean,dat_std):
	return np.array([(feat - feat_mean)/feat_std for feat,feat_mean,feat_std in zip(dat.T,dat_mean,dat_std)]).T	

def remove_unchanging(data):
	dat0 = data[0]
	dat_std = [np.std(feat) for feat in dat0.T]
	#print(dat_std)
	changing = [i for i,item in enumerate(dat_std) if item > 10**-8]
	print(changing)

	data = [dat[:,changing] for dat in data]

	return data,changing

# split data into train and test
def split(data,split_method):
	split_method = split_method
	if split_method == "TIMEWISE":	
		train_share = 0.6
		test_share = 0.2
		train_data = [dat[0:int(np.floor(train_share*np.shape(dat)[0])),:] for dat in data]
		test_data = [dat[int(np.floor(train_share*np.shape(dat)[0])):int(np.floor((train_share+test_share)*np.shape(dat)[0])),:] for dat in data]
	elif split_method == "UNITWISE":
		train_share = 0.2
		test_share = 0.2
		train_data = data[0:int(np.floor(train_share*len(data)))]
		test_data = data[int(np.floor(train_share*len(data))):int(np.floor((train_share+test_share)*len(data)))]		

	return train_data,test_data

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
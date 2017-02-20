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
	try:
		return float(string)
	except ValueError:
		return string

def write_line(line,elemsep):
	return elemsep.join(line)

def filter_wrt(data,key,value):
	return data[data[:,key]==value,:]

explorable_types = ['list','dict','ndarray','void']
def explore(data,branch_limit,expansion_limit,indent=0):
	#print(type(data))
	#print(len(data))
	try:
		print('  '*indent + "{0:s}_{1:d}".format(type(data).__name__,len(data)))
		if len(data) < branch_limit and type(data).__name__ in explorable_types: #[list,dict,type(np.ndarray([])),type(np.void())]:
			if isinstance(data,dict):
				i = 0
				for elem in data:
					if i < expansion_limit:
						explore(data[elem],branch_limit,expansion_limit,indent+1)
					else:
						break
					i += 1
			else:
				i = 0
				for elem in data:
					if i < expansion_limit:
						explore(elem,branch_limit,expansion_limit,indent+1)
					else:
						break
					i += 1	
	except TypeError:
		print('  '*indent + "{0:s}".format(type(data).__name__))

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
		explore(data,11000,10)
			

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
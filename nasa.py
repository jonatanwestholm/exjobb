# nasa.py
# module for exploring datsets: Turbofan, IGBT, Bearing, Mill, CFRP

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import time
import preprocessing as pp
import nasa_explanations

def display_sequential(data,explanations,run_to_fail=False,accumulative=False):
	num_features = np.shape(data[0])[-1]

	max_length = max([np.shape(dat)[0] for dat in data])

	grayscale_range = np.linspace(0.8,0,len(data))

	for i in range(num_features):
		#print(i)
		#print(np.shape(data[0]))
		plt.figure()
		plt.title(explanations[i])
		if run_to_fail:
			plt.xlabel("Cycles before last measurement")
		else:
			plt.xlabel("Cycles since start of measurement")

		j = 0
		for dat in data:
			#print(np.shape(dat))
			if run_to_fail:
				pad = max_length - np.size(dat[:,i])
				dat_ext = np.concatenate((np.nan*np.ones(pad),dat[:,i]))
				plt.plot(list(range(-max_length+1,1)),dat_ext,'--')
			else:
				if accumulative:
					plt.plot(dat[:,i],'--',color=str(grayscale_range[j]))
				else:
					plt.plot(dat[:,i],'--')

			j += 1

	plt.show()
	
def main(args):
	filename = args.filename
	datatype = args.datatype
	dataset = args.dataset.upper()

	if dataset:
		if dataset == "TURBOFAN":
			# uses that the unit measurements come in order in turbofan data
			unit_nbr_location = 0
			data = []
			with open(filename,'r') as f:
				i = 1
				dat = []
				line = f.readline()
				while line:
					#print(line)
					line = pp.read_line(line,args.elemsep,lambda x: float(x))
					#print(line)
					if line[unit_nbr_location] == i:
						dat.append(line)
					else:
						dat = np.array(dat)
						data.append(dat)
						dat = []
						i += 1

					line = f.readline()

			explanations = nasa_explanations.turbofan_features
			run_to_fail = True
			accumulative = False

		elif dataset == "IGBT":
			# collect data
			mat = scipy.io.loadmat(filename)
			data = []
			data.append(list(mat["GATE_VOLTAGE"][0]))
			data.append(list(mat["COLLECTOR_CURRENT"][0]))
			data.append(list(mat["PACKAGE_TEMP"][0]))
			data.append(list(mat["HEAT_SINK_TEMP"][0]))
			data.append(list(mat["COLLECTOR_VOLTAGE"][0]))
			data.append(list(mat["GATE_CURRENT"][0]))

			# package
			data = np.array(data)
			data = data.T
			data = [data]
			#print(data[:5,:3])
			#print(mat['GATE_VOLTAGE'][0][:20])

			explanations = nasa_explanations.igbt_features
			run_to_fail = True
			accumulative = False

		elif dataset == "MILL":
			# collect
			mat = scipy.io.loadmat(filename)
			#print(len(mat['mill'][0]))
			case = int(args.case)
			condition = lambda x: x[0] == case
			tests = pp.filter_wrt_function(mat['mill'][0],condition)
			
			data = []
			for test in tests: #range(len(mat['mill'][0])):
				#print(test[2])
				dat = []
				for j in range(7,13): # collect features 7 to 12
					dat.append(list(np.squeeze(test[j])))

				# package
				dat = np.array(dat)
				dat = dat.T
				data.append(dat)

			#pp.explore(data,200,15)

			# display
			#display_sequential(data,nasa_explanations.mill_features,run_to_fail=False,accumulative=True)
			explanations = nasa_explanations.mill_features
			run_to_fail = False
			accumulative = True


		if __name__ == '__main__':
			if datatype == "SEQUENTIAL":
				display_sequential(data,explanations,run_to_fail,accumulative)
			elif datatype == "PARALLEL":
				if args.max_plots:
					max_plots = int(args.max_plots)
					data = data[:max_plots]
				pp.display_parallel(data,explanations)
		else:
			return data

	else:	
		if datatype == "SEQUENTIAL":
			data = pp.read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines)
			try:
				data.remove([])
			except ValueError:
				pass
			data = [np.array(data)]
			#print(data)
			display_sequential(data)
			
		elif datatype == "LAYERED":
			data = scipy.io.loadmat(filename)
			pp.explore(data,200,50)
			

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',dest = 'filename',default="",help='Name of input file')
    #parser.add_argument('-t','--target',dest = 'target',default="",help='Name of target directory for melting')
    #parser.add_argument('-p','--pattern',dest = 'pattern',default="",help='Input file pattern (regex)')    
    parser.add_argument('-d','--datatype',dest = 'datatype',default="SEQUENTIAL",help='Type of data (e.g. SEQUENTIAL,INSTANCE,LAYERED)')
    parser.add_argument('-s','--dataset',dest = 'dataset',default="",help='Pick dataset (e.g. TURBOFAN,IGBT,BEARING)')
    parser.add_argument('-a','--case',dest = 'case',default=1,help='Case to investigate (Mill)')
    parser.add_argument('-c','--readlines',dest = 'readlines',default="all",help='Number of lines to read')
    parser.add_argument('-e','--elemsep',dest = 'elemsep',default='\t',help='Element Separator')
    parser.add_argument('-l','--linesep',dest = 'linesep',default='\n',help='Line Separator')
    parser.add_argument('-m','--max_plots',dest = 'max_plots',default='',help='Max number of unit plots (only for PARALLEL)')
    args = parser.parse_args()
    main(args)
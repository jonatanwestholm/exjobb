# backblaze.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import preprocessing as pp

explanations = ["Temperature","Humidity","Light","CO2","HumidityRatio"]

def just_the_names(filenames):
	return [filename.split('/')[-1] for filename in filenames]

def main(args):
	filename = args.filename
	datatype = args.datatype

	if datatype == "SEQUENTIAL":
		#pattern = 'PL1331LAGLX6PH.csv'
		pattern = args.pattern

		filenames = glob.glob(filename+pattern)
		names = just_the_names(filenames)
		#print(filename+pattern)
		#print(sorted(filenames,reverse=False))
		print(sorted(filenames,reverse=True))

		feature_idxs = [2,3,4,5,6]
		gt_idx = [7]

		data = [pp.read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines) for filename in sorted(filenames,reverse=True)]
		for dat in data:
			dat.remove([])

		data = [np.array(dat) for dat in data]
		data = [dat[1:,:] for dat in data]

		'''
		for feat in feature_idxs:
			plt.figure()
			for dat in data:
				plt.plot(dat[:,feat])

		plt.show()
		'''

		gt = [dat[:,gt_idx] for dat in data]
		data = [dat[:,feature_idxs] for dat in data]	

		data = pp.normalize_all(data,leave_zero=True)	

		return data,gt,explanations,names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',dest = 'filename',default="",help='Name of input file')
    parser.add_argument('-t','--target',dest = 'target',default="",help='Name of target directory for melting')
    parser.add_argument('-p','--pattern',dest = 'pattern',default="",help='Input file pattern (regex)')    
    parser.add_argument('-d','--datatype',dest = 'datatype',default="SEQUENTIAL",help='Type of data (e.g. SEQUENTIAL,INSTANCE,LAYERED)')
    parser.add_argument('-c','--readlines',dest = 'readlines',default="all",help='Number of lines to read')
    parser.add_argument('-e','--elemsep',dest = 'elemsep',default=',',help='Element Separator')
    parser.add_argument('-l','--linesep',dest = 'linesep',default='\n',help='Line Separator')
    args = parser.parse_args()
    main(args)
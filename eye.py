# eye.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import preprocessing as pp

#explanations = ["EEG {0:d}".format(i) for i in range(14)]
explanations = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
				"O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

def just_the_names(filenames):
	return [filename.split('/')[-1][:-4] for filename in filenames]

def neutralize_outliers(dat):
	for i in range(dat.shape[1]):
		feat = dat[:,i]
		feat_mean = np.mean(feat)
		outliers = np.where(np.abs(feat - feat_mean) > 200)[0]
		dat[outliers,i] = feat_mean

	return dat

def main(args):
	filename = args.filename
	datatype = args.datatype

	if datatype == "SEQUENTIAL":
		#pattern = 'PL1331LAGLX6PH.csv'
		pattern = args.pattern

		filenames = glob.glob(filename+pattern)
		filenames = sorted(filenames,reverse=True)
		names = just_the_names(filenames)
		#print(filename+pattern)
		#print(sorted(filenames,reverse=False))
		print(names)

		feature_idxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
		gt_idx = [14]

		data = [pp.read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines) for filename in sorted(filenames,reverse=True)]
		for dat in data:
			dat.remove([])

		data = [np.array(dat) for dat in data]
		#data = [dat[1:,:] for dat in data]
		data = [neutralize_outliers(dat) for dat in data]

		'''
		gt = [dat[:,gt_idx] for dat in data]
		data = [dat[:,feature_idxs] for dat in data]	
		data = pp.normalize_all(data,leave_zero=True)

		x = np.linspace(0,117,14980)
		print(x.shape)
		for dat in data:
			print(dat.shape)
			plt.figure()
			for feat in feature_idxs:
				plt.plot(x,dat[:,feat])
			plt.plot(x,gt[0]*5,'b')

		plt.xlabel('time / s')
		plt.ylabel('Normalized EEG value')
		plt.title('EEG Eye Features')
		plt.legend(explanations+["Ground Truth"])
		plt.show()
		'''

		'''
		data = pp.normalize_all(data,leave_zero=True)	
		
		for dat in data:
			plt.figure()
			for feat in feature_idxs:
				plt.plot(dat[:,feat])
			gt_array = dat[:,gt_idx]
			gt_array = gt_array - min(gt_array)
			gt_array = gt_array/max(gt_array)
			plt.plot(gt_array)
			plt.title("Occupancy and predictors")
			plt.xlabel("Sample no. (time)")
			plt.ylabel("Value (normalized)")
			plt.legend(explanations+["occupancy"])

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
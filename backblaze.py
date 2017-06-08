# backblaze.py

import argparse
import os
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io
import collections
import glob
import time
import smart_explanations
import preprocessing as pp

BB_SMART_order = [1,2,3,4,5,7,8,9,10,11,12,13,15,22,183,184,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,220,222,223,224,225,226,240,241,242,250,251,252,254,255]
normalized_idx = list(range(5,95,2))
critical_SMART = [5,187,188,197,198]
fail_location = 4 # column where failure is reported

#min_sample_time = 30

def dead_rows(data,filenames):
	for dat,filename in zip(data,filenames):
		for i,row in enumerate(dat):
			if len(np.where(np.isfinite(row))[0]) == 0:
				print("Unit {0:s}, row {1:d}".format(filename,i))

def just_the_names(filenames):
	return [filename.split('/')[-1][:-4] for filename in filenames]

def all_smart_except(remove):
	#N = np.shape(data[0])[1]
	#print(data[0][:,0])
	idxs = [i for i,smart in enumerate(BB_SMART_order) if not smart in remove]
	return idxs

def smart_expl(idx):
	try:
		key = BB_SMART_order[idx]
		expl = "{0:d}: {1:s}".format(key,smart_explanations.smart[BB_SMART_order[idx]])
	except KeyError:
		expl = "S.M.A.R.T feature " + str(BB_SMART_order[idx])
	return key,expl

def caught_failure(dat,name,qualified):
	if not "_fail" in name:
		return False

	critical_idxs = [qualified.index(i) for i in critical_SMART if i in qualified]
	final_values = dat[-1,critical_idxs]
	#print(final_values)

	critical_values = np.sum(final_values[:-2] < 100)
	#print(final_values[:-2] < 100)
	#assumes that 197 and 198 are reported. Checks out with data
	critical_values += np.any(final_values[-2:] < 100) # because 197 and 198 are very correlated
	#print(final_values[-2:] < 100)
	#print(critical_values)

	return critical_values >= 1

def remove_caught_failures(data,names,qualified):
	no_predicted_fail = [i for i in range(len(data)) if not caught_failure(data[i],names[i],qualified)]
	return [data[i] for i in no_predicted_fail], no_predicted_fail

def melt_instance(args,dir_name,pattern,serial_location,model_location):
	target = args.target
	filenames = glob.glob(dir_name+pattern)
	print(dir_name+pattern)
	print(filenames)
	
	name_pat = args.pattern

	for filename in sorted(filenames):
		f = open(filename,'r').read()
		lines = f.split(args.linesep)

		for line in lines:
			line = pp.read_line(line,args.elemsep,lambda x: x)
			if not line:
				continue
			#print(line[location])
			#if line[location] == serial_number:
			#	found = True
			#	break
			#print('found one!')

			serial_number = line[serial_location]
			model_number = line[model_location]
			name = model_number+"_"+serial_number
			name = name.replace(" ","")
			if re.findall(name_pat,name):
				#print(serial_number)
				#print(target+serial_number+'.csv')
				f = open(target+name+'.csv','a')
				f.write(pp.write_line(line + [args.linesep],args.elemsep))
				f.close()

				if line[fail_location] == '1':
					os.rename(target+name+'.csv',target+name+'_fail.csv')
			else:
				pass #print(name)

def product_model_dictionary(filenames):
	d = collections.defaultdict(lambda : [0,0])
	for fname in filenames:
		prodname = fname.split('_')[0]
		count = d[prodname]
		count[0] += 1
		if "_fail" in fname:
			count[1] += 1

	return d

def product_failures(args,filename):
	filenames = glob.glob(filename+"*.csv")
	filenames = [fname.split('/')[-1] for fname in filenames]
	filenames = [fname[:-4] for fname in filenames]
	d = product_model_dictionary(filenames)
	for key in d:
		total = d[key][0]
		failed = d[key][1]
		share = failed/total
		print(key.ljust(30,' ')+"Total: {0:d}".format(total).ljust(15,' ')+
								"Failed: {0:d}".format(failed).ljust(15,' ')+
								"Share: {0:.2f}".format(share).ljust(15,' '))

def main(args):
	filename = args.filename
	datatype = args.datatype

	if datatype == "SEQUENTIAL":
		#pattern = 'PL1331LAGLX6PH.csv'
		pattern = args.pattern

		filenames = glob.glob(filename+pattern)
		#print(filename+pattern)
		#print(filenames)

		data = [pp.read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines) for filename in filenames]
		for dat in data:
			dat.remove([])
		data = [np.array(dat) for dat in data]

		#numeric_per_row = [pp.count_numeric(row) for row in data[0]]
		#print("Max numeric: " + str(max(numeric_per_row)))
		#print("Argmax numeric: " + str(np.argmax(numeric_per_row)))

		if __name__ == '__main__':
			names = just_the_names(filenames)
			# Data selection
			data = [dat[:,normalized_idx] for dat in data]
			#dead_rows(data,filenames)
			qualified = BB_SMART_order
			#data,__ = remove_caught_failures(data,names,qualified)

			max_length = max(map(lambda x: np.shape(x)[0],data))

			x_range = list(range(-max_length+1,1))
			#print(max_length)
			#return

			#data = filter_wrt(data,0,2)

			#while True:
			#	x = input('Which feature do you want to look at? ')
			#	x = int(x)
			i = 0
			for x in range(data[0].shape[1]):
				plotted = False
				
				for dat,filename in zip(data,filenames):
					#print(dat)
					#if i == 45:
					#	print("xx" + dat[0,x] + "xx")
					if np.isnan(dat[0,x]):
						#print('empty feature at {0:d}'.format(x))
						continue
					else:
						if not plotted:
							plt.figure()
							plt.xlabel('Days before failure (red) or end of measurement (blue)')
							try:
								plt.title("S.M.A.R.T feature {0:d}: {1:s}".format(BB_SMART_order[i],smart_expl(i)[1]))
							except IndexError:
								plt.title("Plot {0:d}".format(i))
							except KeyError:
								plt.title("S.M.A.R.T feature {0:d}".format(BB_SMART_order[i]))
							plotted = True
					#y = input('Which feature do you want to look at? ')
					#y = int(y)

					#for y in range(10):
					#plt.plot(filter_wrt(data,0,y)[:,x])
					pad = max_length - np.size(dat[:,x])
					dat_ext = np.concatenate((np.nan*np.ones(pad),dat[:,x]))
					if re.findall("_fail",filename):
						plt.plot(x_range,dat_ext,'r',linewidth=5)
					else:	
						#print(filename)
						#print(re.match("_fail",filename))
						plt.plot(x_range,dat_ext,'b--')		

				if not plotted:
					print("nothing to plot for {0:d}".format(i))
					#plt.close()
				i += 1
			plt.show()

		else:

			names = just_the_names(filenames)
			# Data selection
			data = [dat[:,normalized_idx] for dat in data]
			dead_rows(data,names)

			idxs = set(all_smart_except([194]))
			print(idxs)
			#print(set(pp.numeric_idxs(data)))
			idxs = set.intersection(idxs,set(pp.numeric_idxs(data)))
			print(idxs)
			#print(pp.changing_idxs(data))
			idxs = set.intersection(idxs,set(pp.changing_idxs(data)))
			print(idxs)
			print("Qualified indexes: " + str(sorted(idxs)))
			print("Qualified indexes: " + str(sorted([BB_SMART_order[i] for i in idxs])))
			qualified = sorted([BB_SMART_order[i] for i in idxs])
			
			data = [dat[:,sorted(idxs)] for dat in data]
			keys = [smart_expl(i)[0] for i in sorted(idxs)]
			explanations = [smart_expl(i)[1] for i in sorted(idxs)]

			print("before removing missing, small, and predicted failures: "+ str(len(data)))
			__, no_missing = pp.remove_instances_with_missing(data)
			min_sample_time = args.settings["failure_horizon"]+70
			__, no_small = pp.remove_small_samples(data,min_sample_time)
			#print(no_small)
			if 1:
				__, no_predicted_failures = remove_caught_failures(data,names,qualified)
			else:
				no_predicted_failures = list(range(len(data)))
			cleared_idxs = set.intersection(set(no_missing),set(no_small),set(no_predicted_failures))
			cleared_idxs = list(cleared_idxs)
			#print(cleared_idxs)
			data = [data[idx] for idx in sorted(cleared_idxs)]
			#print([len(dat) for dat in data])
			names = [names[idx] for idx in sorted(cleared_idxs)]
			print("after removing missing, small, and predicted failures: "+ str(len(data)))

			# Mathematical preprocessing
			# add 0 to beginning
			#num_features = len(idxs)
			#data = [np.concatenate([np.zeros([1,num_features]),dat],axis=0) for dat in data]

			extended_features = False
			if extended_features:		
				exta = pp.differentiate(data)
				#data = pp.smooth(data,5)
				exta = pp.filter(exta,np.array([1]),np.array([1,-0.8]))
				data = [dat[1:,:] for dat in data] # have to take a away first so that lengths are correct
				#print(exta[0].shape)
				#print(data[0].shape)
				data = [np.concatenate([dat,ext],axis=1) for dat,ext in zip(data,exta)]

				expl_ext = [expl + " (modified)" for expl in explanations]
				explanations = explanations + expl_ext
			else:
				pass
				#data = pp.differentiate(data)
				#sdata = pp.smooth(data,5)
				#data = pp.filter(data,np.array([1]),np.array([1,-0.8]))

			data = pp.normalize_all(data,leave_zero=True,mean_def=100)
			#print(explanations)
			#print(keys)
			print("Explanations " + " ".join(["{0:s}: {1:s}".format(str(key),str(explanation)) for key, explanation in zip(keys,explanations)]))

			if args.test_type == "PREDICTION":
				gt = []
			elif args.test_type in ["CLASSIFICATION","REGRESSION"]:
				X = []
				Y = []
				failed = ["_fail" in name for name in names]

				for x,y in pp.impending_failure(data,failed,args.settings["failure_horizon"],"CLASSIFICATION"):
					X.append(x)
					Y.append(y)

				#order = np.random.choice(len(X),len(X),replace=False)
				#data = [X[ord_i] for ord_i in order]
				#gt = [Y[ord_i] for ord_i in order]
				#data = X
				gt = Y

			return data,gt,explanations,names
			
	elif datatype == "INSTANCE":
		pattern = '[0-9-]*.csv'
		#pattern = args.pattern
		#serial_number = "MJ0351YNG9Z0XA"
		serial_location = 1
		model_location = 2

		melt_instance(args,filename,pattern,serial_location,model_location)
	elif datatype == "POPULATION_STATISTICS":
		product_failures(args,filename)

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
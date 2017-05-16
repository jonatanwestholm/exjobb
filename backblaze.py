# backblaze.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import time
import smart_explanations
import preprocessing as pp

BB_SMART_order = [1,2,3,4,5,7,8,9,10,11,12,13,15,22,183,184,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,220,222,223,224,225,226,240,241,242,250,251,252,254,255]
normalized_idx = list(range(5,95,2))

min_sample_time = 30

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
		expl = smart_explanations.smart[BB_SMART_order[idx]]
	except KeyError:
		expl = "S.M.A.R.T feature " + str(BB_SMART_order[idx])
	return expl

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

				fail_location = 4 # column where failure is reported
				if line[fail_location] == '1':
					os.rename(target+name+'.csv',target+name+'_fail.csv')
			else:
				pass #print(name)

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
			max_length = max(map(lambda x: np.shape(x)[0],data))

			x_range = list(range(-max_length+1,1))
			#print(max_length)
			#return

			#data = filter_wrt(data,0,2)

			#while True:
			#	x = input('Which feature do you want to look at? ')
			#	x = int(x)
			i = 0
			for x in range(5,np.shape(data[0])[1],2):
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
								plt.title("S.M.A.R.T feature {0:d}: {1:s}".format(BB_SMART_order[i],smart_expl(i)))
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
			# Data selection
			data = [dat[:,normalized_idx] for dat in data]
			dead_rows(data,filenames)

			idxs = set(all_smart_except([194]))
			print(idxs)
			#print(set(pp.numeric_idxs(data)))
			idxs = set.intersection(idxs,set(pp.numeric_idxs(data)))
			print(idxs)
			#print(pp.changing_idxs(data))
			idxs = set.intersection(idxs,set(pp.changing_idxs(data)))
			print(idxs)
			
			data = [dat[:,sorted(idxs)] for dat in data]
			explanations = [smart_expl(i) for i in sorted(idxs)]

			print("before removing missing and small: "+ str(len(data)))
			data,__ = pp.remove_instances_with_missing(data)
			data = pp.remove_small_samples(data,min_sample_time)
			print("after removing missing and small: "+ str(len(data)))

			# Mathematical preprocessing
			# add 0 to beginning
			num_features = len(idxs)
			data = [np.concatenate([np.zeros([1,num_features]),dat],axis=0) for dat in data]

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
				data = pp.differentiate(data)
				#sdata = pp.smooth(data,5)
				#data = pp.filter(data,np.array([1]),np.array([1,-0.8]))

			data = pp.normalize_all(data,leave_zero=True)
			print("Qualified indexes: " + str(sorted(idxs)))
			print("Explanations: " + str(explanations))

			names = just_the_names(filenames)

			if args.test_type == "PREDICTION":
				gt = []
			elif args.test_type in ["CLASSIFICATION","REGRESSION"]:
				X = []
				Y = []
				failed = ["_fail" in name for name in names]

				for x,y in pp.impending_failure(data,failed,args.settings["failure_horizon"],"CLASSIFICATION"):
					X.append(x)
					Y.append(y)

				data = X
				gt = Y

			return data,gt,explanations,names
			
	elif datatype == "INSTANCE":
		pattern = '[0-9-]*.csv'
		#pattern = args.pattern
		#serial_number = "MJ0351YNG9Z0XA"
		serial_location = 1
		model_location = 2

		melt_instance(args,filename,pattern,serial_location,model_location)

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
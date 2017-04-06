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

def dead_rows(data,filenames):
	for dat,filename in zip(data,filenames):
		for i,row in enumerate(dat):
			if len(np.where(np.isfinite(row))[0]) == 0:
				print("Unit {0:s}, row {1:d}".format(filename,i))

def just_the_names(filenames):
	return [filename.split('/')[-1] for filename in filenames]

def condense_name(filename):
	return str([])

def melt_instance(args,dir_name,pattern,location):
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

			serial_number = line[location]
			serial_number = just_the_names([serial_number])[0]
			if re.findall(name_pat,serial_number):
				#print(serial_number)
				#print(target+serial_number+'.csv')
				f = open(target+serial_number+'.csv','a')
				f.write(pp.write_line(line + [args.linesep],args.elemsep))
				f.close()

			else:
				print(serial_number)

def main(args):
	filename = args.filename
	datatype = args.datatype

	if datatype == "SEQUENTIAL":
		#pattern = 'PL1331LAGLX6PH.csv'
		pattern = args.pattern

		filenames = glob.glob(filename+pattern)
		#print(filename+pattern)
		#print(filenames)

		data = [pp.read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines,mapping=lambda x: x) for filename in filenames]
		for dat in data:
			dat.remove([])

		name_location = 1
		stream_count_location = 3

		for dat,filename in zip(data,filenames):
			try:
				stream_stats = np.array([int(row[stream_count_location]) for row in dat])
			except ValueError:
				print(filename)
				stream_stats = [0]

			name = dat[0][name_location]

			if np.max(stream_stats) > 3e+6:
				plt.figure()
				plt.title(name)
				plt.axis([0, 40, 0, 1e+7])
				plt.plot(stream_stats)

		plt.show()

		#numeric_per_row = [pp.count_numeric(row) for row in data[0]]
		#print("Max numeric: " + str(max(numeric_per_row)))
		#print("Argmax numeric: " + str(np.argmax(numeric_per_row)))
			
	elif datatype == "INSTANCE":
		pattern = '[a-z0-9-]*.csv'
		#pattern = args.pattern
		#serial_number = "MJ0351YNG9Z0XA"
		location = 4

		melt_instance(args,filename,pattern,location)	

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
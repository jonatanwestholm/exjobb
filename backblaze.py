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

BB_SMART_order = [1,2,3,4,5,7,9,10,11,12,13,15,22,183,184,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,220,222,223,224,225,226,240,241,242,250,251,252,254,255]

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
			if re.findall(name_pat,serial_number):
				#print(serial_number)
				#print(target+serial_number+'.csv')
				f = open(target+serial_number+'.csv','a')
				f.write(pp.write_line(line + [args.linesep],args.elemsep))
				f.close()

				fail_location = 4
				if line[fail_location] == '1':
					os.rename(target+serial_number+'.csv',target+serial_number+'_fail.csv')
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

		data = [pp.read_file(filename,elemsep=args.elemsep,linesep=args.linesep,readlines=args.readlines) for filename in filenames]
		for dat in data:
			dat.remove([])
		data = [np.array(dat) for dat in data]

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
			plt.figure()
			plt.xlabel('Days before failure (red) or end of measurement (blue)')
			try:
				plt.title("S.M.A.R.T feature {0:d}: {1:s}".format(BB_SMART_order[i],smart_explanations.smart[BB_SMART_order[i]]))
			except IndexError:
				plt.title("Plot {0:d}".format(i))
			except KeyError:
				plt.title("S.M.A.R.T feature {0:d}".format(BB_SMART_order[i]))
			
			for dat,filename in zip(data,filenames):
				#print(dat)
				#if i == 45:
				#	print("xx" + dat[0,x] + "xx")
				if dat[0,x] == '':
					#print('empty feature at {0:d}'.format(x))
					continue
				else:
					plotted = True
				#y = input('Which feature do you want to look at? ')
				#y = int(y)

				#for y in range(10):
				#plt.plot(filter_wrt(data,0,y)[:,x])
				pad = max_length - np.size(dat[:,x])
				dat_ext = np.concatenate((np.nan*np.ones(pad),dat[:,x]))
				if re.findall("_fail",filename):
					plt.plot(x_range,dat_ext,color='r')
				else:	
					#print(filename)
					#print(re.match("_fail",filename))
					plt.plot(x_range,dat_ext,color='b')		

			if not plotted:
				print("nothing to plot for {0:d}".format(i))
				plt.close()
			i += 1
		plt.show()
	elif datatype == "INSTANCE":
		pattern = '[0-9-]*.csv'
		#pattern = args.pattern
		#serial_number = "MJ0351YNG9Z0XA"
		location = 1

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
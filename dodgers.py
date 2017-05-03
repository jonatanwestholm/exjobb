# backblaze.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob

import models_auxilliary as mod_aux
import preprocessing as pp

buf = 3000
name = ""

class Time_interval:
	def __init__(self,date,start_time,end_time): #assumes that intervals don't stretch over midnight
		self.start_time = self.parse_datetime(date,start_time)
		self.end_time = self.parse_datetime(date,end_time)

	def parse_datetime(self,date,time):
		date = self.parse_date(date)
		return date+self.parse_time(time)

	def parse_date(self,date): # not correct but preserves order among dates
		month,day,year = map(int,date.split("/"))
		year = year % 100
		return 86400*(366*year+31*month+day)

	def parse_time(self,timestr):
		try:
			hour,minute,second = map(int,timestr.split(":"))
		except ValueError:
			hour,minute = map(int,timestr.split(":"))
			second = 0
		return 3600*hour+60*minute+second

	def is_inside(self,date,time):
		time = self.parse_datetime(date,time)
		#return time >= self.start_time - buf and time <= self.end_time + buf
		if name == "Dodgers":
			return time >= self.end_time and time <= self.end_time + buf
		else:
			return time >= self.start_time and time <= self.end_time

	def is_after(self,date,time):
		time = self.parse_datetime(date,time)
		if name == "Dodgers":
			return time > self.end_time + buf
		else:
			return time > self.end_time

def test_time_interval():
	event_date = "07/26/05"
	start_time = "11:00:00"
	end_time = "14:00:00"

	dates = ["07/24/05","07/26/05","07/26/05"]
	times = ["00:00:00","13:30:00","15:30:00"]

	event = Time_interval(event_date,start_time,end_time)

	for date,time in zip(dates,times):
		print(event.is_inside(date,time))
		print(event.is_after(date,time))

def generate_gt(data,event_times):
	i = 0
	N = len(event_times)
	event = event_times[i]
	gt = []
	for dat in data:
		date = dat[0]
		time = dat[1]
		if event.is_inside(date,time):
			gt.append(1)
		else:
			gt.append(0)

		if event.is_after(date,time):
			i += 1
			if i >= N:
				event = event_times[-1]
			else:
				event = event_times[i]
				
	gt = np.array(gt)
	gt = gt.reshape([len(gt),1])
	return gt

def make_cycle_data(dat,T):
	N = len(dat)
	N = N - N % T
	num_cycles = int(N/T)

	dat = dat[:N].reshape([num_cycles,T])

	return dat

def main_linear_subspace(dat,coverage):
	__,S,V = np.linalg.svd(dat,full_matrices=False)
	cum_rel = mod_aux.cumulative_singular_values(S,plot=False)
	if coverage >= 1:
		k = coverage
	else:
		k = min(np.where(cum_rel>coverage)[0])
	print("subspace dimensionality: " +str(k))

	return V[:k,:]

def lagged_features(intensity,lag):
	features = np.zeros([len(intensity)-lag,lag])
	for i in range(lag):
		features[:,i] = intensity[i:-(lag-i)]

	return features

def main(args):
	filename = args.filename
	names = pp.just_the_names([filename])
	print(names)
	global name
	name = names[0]

	data_filename = filename+".data"
	event_filename = filename+".events"

	data = pp.read_file(data_filename,elemsep=",",linesep="\n",readlines="all",mapping= lambda x: x)
	events = pp.read_file(event_filename,elemsep=",",linesep="\n",readlines="all",mapping= lambda x: x)

	#data.remove([])
	data = list(filter(([]).__ne__,data))

	if names[0] == "Dodgers":
		intensity = np.array([int(row[2]) for row in data])
		data = [row for row,ints in zip(data,intensity) if ints != -1]
		T = 288
		num_subspace = 4
	elif names[0] == "CalIt2":
		for row in data:
			if len(row) < 4:
				print(row)
		intensity = np.array([int(row[3]) for row in data])
		data = [row[1:] for row,ints in zip(data,intensity) if ints != -1]
		intensity = intensity
		T = 48
		num_subspace = 3


	intensity = intensity[np.where(intensity!=-1)[0]]

	#intensity = intensity.reshape([len(intensity),1])
	#intensity = pp.smooth(intensity.T,2)[0].T

	raw_intensity = intensity
	
	event_times = [Time_interval(event[0],event[1],event[2]) for event in events]
	gt = generate_gt(data,event_times)

	intensity = intensity/np.std(intensity) #pp.normalize(intensity,leave_zero=True)

	#intensity = pp.remove_harmonic_trend(intensity,T)
	#intensity = intensity.reshape([len(intensity),1])
	#intensity = pp.smooth(intensity,20)

	X = make_cycle_data(intensity,T)
	Cs = main_linear_subspace(X,num_subspace)
	#print(X.shape)
	#print(Cs.shape)
	X = X - np.dot(np.dot(X,Cs.T),Cs)
	numel = np.prod(X.shape)
	intensity = X.reshape([numel,1])
	intensity = pp.smooth(intensity,1)
	
	if args.model == "ESN":
		lag = 1
	elif args.model == "SVM":
		lag = 5
	elif args.model == "MLP":
		lag = 5
	intensity = lagged_features(intensity,lag)
	gt = gt[lag:]	
	
	'''
	print(gt.shape)
	plt.plot(intensity[:,-1])
	plt.plot(gt)
	plt.show()	
	'''

	explanations = ["intensity"]

	if __name__ == '__main__':
		plt.figure()
		plt.plot(raw_intensity[::2]/10,"b")
		plt.plot(raw_intensity[1::2]/10,"k")
		plt.plot(intensity[::2],"g")
		plt.plot(gt[::2],"r")
		plt.legend(["Raw intensity out flow", "Raw intensity in flow", "Processed intensity", "Event at stadium"])
		plt.title("{0:s}".format(names[0]))
		plt.xlabel("Sample no. (time)")
		plt.ylabel("No. cars/ 5 secs")
		plt.show()

	return [intensity],[gt],explanations,names

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
    args.model = "ESN"
    main(args)
    #test_time_interval()
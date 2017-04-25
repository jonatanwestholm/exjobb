# backblaze.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import preprocessing as pp

class Time_interval:
	def __init__(self,date,start_time,end_time): #assumes that intervals don't stretch over midnight
		self.start_time = self.parse_datetime(date,start_time)
		self.end_time = self.parse_datetime(date,end_time)

	def parse_datetime(self,date,time):
		date = self.parse_date(date)
		return date+self.parse_time(time)

	def parse_date(self,date): # not correct but preserves order among dates
		day,month,year = map(int,date.split("/"))
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
		return time >= self.start_time and time <= self.end_time

	def is_after(self,date,time):
		time = self.parse_datetime(date,time)
		return time > self.end_time

def generate_gt(data,event_times):
	i = 0
	N = len(event_times)
	event = event_times[i]
	gt = []
	for dat in data:
		#print(dat)
		date = dat[0]
		time = dat[1]
		if event.is_inside(date,time):
			gt.append(1)
		else:
			gt.append(0)

		if event.is_after(date,time):
			i += 1
			if i == N:
				break
			else:
				event_times[i]
				
	gt = np.array(gt)
	gt = gt.reshape([len(gt),1])
	return gt

def main(args):
	filename = args.filename
	names = pp.just_the_names([filename])
	print(names)

	data_filename = filename+".data"
	event_filename = filename+".events"

	data = pp.read_file(data_filename,elemsep=",",linesep="\n",readlines="all",mapping= lambda x: x)
	events = pp.read_file(event_filename,elemsep=",",linesep="\n",readlines="all",mapping= lambda x: x)

	if names[0] == "Dodgers":
		intensity = np.array([int(row[2]) for row in data])
	elif names[0] == "CalIt2":
		intensity = np.array([int(row[3]) for row in data])

	data = [row for row,ints in zip(data,intensity) if ints != -1]
	intensity = intensity[np.where(intensity!=-1)[0]]
	
	event_times = [Time_interval(event[0],event[1],event[2]) for event in events]
	gt = generate_gt(data,event_times)

	intensity = intensity/np.std(intensity) #pp.normalize(intensity,leave_zero=True)
	intensity = intensity.reshape([len(intensity),1])
	

	explanations = ["intensity"]

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
    main(args)
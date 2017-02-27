# subgroup.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob

import models as Models
import nasa
import backblaze

# returns data = [np.array()]
def preprocess(args):
	dataset = args.dataset.upper()

	if dataset in ["TURBOFAN","MILL","IGBT"]:
		data = nasa.main(args)
	elif dataset == "BACKBLAZE":
		data = backblaze.main(args)
	elif dataset == "ARMA_SIM":
		data = sim.arma_sim(np.array([1]),np.array([1,0.5,-0.2]),1000,num=5)
	else:
		print("No matching dataset!")

	return data

def split(args,data):
	split_method = args.split_method.lower()
	# split data into train and test
	if split_method == "timewise":	
		train_share = 0.6
		test_share = 0.2
		train_data = [dat[0:int(np.floor(train_share*np.shape(dat)[0])),:] for dat in data]
		test_data = [dat[int(np.floor(train_share*np.shape(dat)[0])):int(np.floor((train_share+test_share)*np.shape(dat)[0])),:] for dat in data]
	elif split_method == "unitwise":
		train_share = 0.6
		test_share = 0.2
		train_data = data[0:int(np.floor(train_share*len(data)))]
		test_data = data[int(np.floor(train_share*len(data))):int(np.floor((train_share+test_share)*len(data)))]

	return train_data,test_data

## Training

# settings = {'type': 'type of learning model'}
# returns models = [obj]
def train(data,settings):
	train_type = settings['type']

	if train_type == "TRIVIAL":
		models = [Models.Trivial(list(range(np.shape(data[0])[1])))]

	return models

## Testing

def reset_models(models):
	for mod in models:
		mod.reset()

def update_models(data,models):
	for mod in models:
		mod.update_array(data[:,mod.subgroup])

def predict_data(dat,models,k):
	pred_mat = np.zeros_like(dat)
	i = 0
	for sample in dat:
		#print(sample)
		pred_array = np.zeros_like(sample)
		for mod in models:
			mod.update(sample[mod.subgroup])
			pred_array[mod.subgroup] = mod.predict(k)

		pred_mat[i,:] = pred_array
		i += 1
	return pred_mat

# settings = {'type': 'type of test','k': 'prediction length','err_measure':'error measure'}
# returns either: predicted series or predicted classes/scores
def test(train_data,test_data,models,settings,lables=[]):
	args = settings['args']
	test_type = settings['type']
	k = settings['k']
	split_method = args.split_method

	pred = []
	if test_type == "PREDICTION":
		# take prediction unit by unit - units are assumed to be independent
		for tr_dat,test_dat in zip(train_data,test_data):
			# reset all inner states first
			reset_models(models)

			# set states of models
			if split_method == "timewise":
				update_models(tr_dat,models)

			pred.append(predict_data(test_dat,models,k))

	elif test_type == "CLASSIFICATION":
		pass # how to do this with subgroups? not my problem for now

	return pred

# outputs performance scores, plots
def evaluate(pred,gt):
	rmses = []
	for pred_mat,gt_mat in zip(pred,gt):
		#print(pred_mat)
		fro = np.linalg.norm(pred_mat-gt_mat)
		rms = fro/np.sqrt(np.size(pred_mat))
		rmses.append(rms)
		print("RMS norm of error: {0:.3f}".format(rms))
		'''
		for i in range(np.shape(pred_mat)[1]):
			plt.figure()
			plt.plot(pred_mat[:,i],'r')
			plt.plot(gt_mat[:,i],'b')
		plt.show()
		'''
	rmses = np.array(rmses)
	print("Avg: {0:.3f}, Min: {1:.3f}, Max: {2:.3f}".format(np.mean(rmses),np.min(rmses),np.max(rmses)))

	return 0

def subgroup_learn(data,args):
	train_data,test_data = split(args,data)

	train_settings = {'type': args.model}
	models = train(train_data,train_settings)

	pred_k = int(args.pred_k)
	test_settings = {'type': args.test_type,'args': args,'k':pred_k}
	#
	pred = test(train_data,[test_dat[:-pred_k] for test_dat in test_data],models,test_settings)
	gt = [test_dat[pred_k:] for test_dat in test_data]

	return evaluate(pred,gt)

def main(args):
	data = preprocess(args)

	# candidate generation
	subgroup = data

	# subgroup learning
	subgroup_learn(subgroup,args)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--filename',dest = 'filename',default="",help='Name of input file')
	#parser.add_argument('-t','--target',dest = 'target',default="",help='Name of target directory for melting')
	parser.add_argument('-p','--pattern',dest = 'pattern',default="",help='Input file pattern (regex)')
	parser.add_argument('-a','--datatype',dest = 'datatype',default="SEQUENTIAL",help='Type of data (e.g. SEQUENTIAL,INSTANCE,LAYERED)')
	parser.add_argument('-d','--dataset',dest = 'dataset',default="",help='Pick dataset (e.g. TURBOFAN,IGBT,BEARING,MILL)')
	parser.add_argument('--case',dest = 'case',default=1,help='Case to investigate (Mill)')
	parser.add_argument('-c','--readlines',dest = 'readlines',default="all",help='Number of lines to read')
	parser.add_argument('-e','--elemsep',dest = 'elemsep',default='\t',help='Element Separator')
	parser.add_argument('-l','--linesep',dest = 'linesep',default='\n',help='Line Separator')

	parser.add_argument('-m','--model',dest = 'model',default="TRIVIAL",help='Prediction model')
	parser.add_argument('-k','--pred_k',dest = 'pred_k',default=1,help='How many steps of the future to predict')
	parser.add_argument('-s','--split_method',dest = 'split_method',default="timewise",help='How to split data into training and test? (timewise,unitwise)')
	parser.add_argument('-t','--test_type',dest = 'test_type',default="PREDICTION",help='Type of learning')
	
	args = parser.parse_args()
	main(args)
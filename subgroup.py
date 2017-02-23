# subgroup.py

import argparse
import os
import re
import numpy as np
import glob

import models
import nasa
import backblaze

# returns data = [np.array()]
def preprocess(args):
	dataset = args.dataset.upper()

	if dataset in ["TURBOFAN","MILL","IGBT"]:
		data = nasa.main(args)
	elif dataset == "BACKBLAZE":
		data = backblaze.main(args)
	else:
		print("No matching dataset!")

	return data

def split(args,data):
	split_method = args.split_method
	# split data into train and test
	if split_method == "timewise":	
		train_share = 0.8
		test_share = 0.2
		train_data = [dat[0:np.floor(train_share*np.shape(dat)[0]),:] for dat in data]
		test_data = [dat[np.floor(train_share*np.shape(dat)[0]):np.floor((train_share+test_share)*np.shape(dat)[0]),:] for dat in data]
	elif split_method == "unitwise":
		train_share = 0.8
		test_share = 0.2
		train_data = data[0:np.floor(train_share*len(data))]
		test_data = data[np.floor(train_share*len(data)):np.floor((train_share+test_share)*len(data))]

	return train_data,test_data

# settings = {'type': 'type of learning model'}
# returns models = [obj]
def train(data,settings):
	train_type = settings['type']

	if train_type == "TRIVIAL":
		models = [models.Trivial(list(range(np.shape(data[0])[1])))]

	return models

# settings = {'type': 'type of test','k': 'prediction length','err_measure':'error measure'}
# returns either: predicted series or predicted classes/scores
def test(train_data,test_data,models,settings,lables=[]):
	args = settings['args']
	split_method = args.split_method

	for mod in models:
		mod.reset()

	pred = []
	test_type = settings['type']
	if test_type == "PREDICTION":
		k = settings['k']
		# take prediction unit by unit - units are assumed to be independent
		for tr_dat,test_dat in zip(train_data,test_data):
			# set states of models
			if split_method == "timewise":
				for mod in models:
					mod.update(tr_dat[:,mod.subgroup])

			pred_mat = np.zeros_like(test_dat)
			i = 0
			for sample in test_dat:
				pred_array = np.zeros_like(sample)
				for mod in models:
					pred_array[mod.subgroup] = mod.predict(k)
					mod.update(sample[mod.subgroup])

				pred_mat[i,:]
				i += 1

			pred.append(pred_mat)
	elif test_type == "CLASSIFICATION":
		pass # how to do this with subgroups? not my problem for now

	return pred

# outputs performance scores, plots
def evaluate(pred,gt):
	for pred_mat,gt_mat in zip(pred,gt):
		print("Frobenius norm of error: {0:.3f}".format(np.linalg.norm(pred_mat-gt_mat)))

def main(args):
	data = preprocess(args)

	train_data,test_data = split(args,data)

	train_settings = {'type': 'TRIVIAL'}
	models = train(train_data,train_settings)

	pred_k = 1
	test_settings = {'type': 'PREDICTION','args': args,'k':pred_k}
	pred = test(train_data,test_data[:-1],models,test_settings)
	gt = test_data[1:]

	evaluate(pred,gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',dest = 'filename',default="",help='Name of input file')
    #parser.add_argument('-t','--target',dest = 'target',default="",help='Name of target directory for melting')
    parser.add_argument('-p','--pattern',dest = 'pattern',default="",help='Input file pattern (regex)')    
    #parser.add_argument('-d','--datatype',dest = 'datatype',default="SEQUENTIAL",help='Type of data (e.g. SEQUENTIAL,INSTANCE,LAYERED)')
    parser.add_argument('-s','--dataset',dest = 'dataset',default="",help='Pick dataset (e.g. TURBOFAN,IGBT,BEARING)')
    parser.add_argument('-a','--case',dest = 'case',default=1,help='Case to investigate (Mill)')
    parser.add_argument('-c','--readlines',dest = 'readlines',default="all",help='Number of lines to read')
    parser.add_argument('-e','--elemsep',dest = 'elemsep',default='\t',help='Element Separator')
    parser.add_argument('-l','--linesep',dest = 'linesep',default='\n',help='Line Separator')
    args = parser.parse_args()
    main(args)
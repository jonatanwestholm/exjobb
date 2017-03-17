# subgroup.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

import subgroup_auxilliary as aux
import preprocessing as pp
import sim
import models as Models
import nasa
import backblaze

# auxilliary

# class's tasks:
# -control that no feature is modelled more than once by subgroups
# -keep track of which features are not modelled by subgroups

## Preprocessing

# returns data = [np.array()]
# preprocessing is dataset specific so it's handled in separate scripts
def preprocess(args):
	dataset = args.dataset.upper()

	if dataset in ["TURBOFAN","MILL","IGBT"]:
		data,explanations = nasa.main(args)
	elif dataset == "BACKBLAZE":
		data,explanations = backblaze.main(args)
	#elif dataset == "ARMA_SIM":
	#	data = sim.arma_sim(np.array([1]),np.array([1,0.5,-0.2]),1000,num=5)
	elif dataset == "VARMA_SIM":
		if args.filename:
			data = sim.read(args)
			#data = [pp.normalize(dat) for dat in data]
		else:
			num_timepoints = args.settings["num_timepoints"]
			num_samples = args.settings["num_samples"]
			case = args.settings["case"]
			data = [sim.mixed_varma(num_timepoints,case) for i in range(num_samples)]
			sim.write(data,args)

		N = aux.num_features(data[0])
		explanations = ["feature {0:d}".format(i) for i in range(N)]
	else:
		print("No matching dataset!")

	args.explanations = explanations

	return data

## Candidate Generation

def candidate_generate(train_data,remaining,num,args):
	N = len(remaining)

	gen = args.generation.upper()
	if gen == "TRIVIAL":
		cand = list(range(N))
		cands = [aux.map_idx(remaining,cand)]*num # subgroups won't be edited, only read
	elif gen == "RANDOM":
		# take out random 
		cands = []
		for i in range(num):
			while True:
				# generate random subset
				cand = list(np.where(np.random.randint(0,2,N))[0])
				length = len(cand)
				if args.settings["min_subgroup_length"] <= length and length <= args.settings["max_subgroup_length"]:
					#if cand == [1,13,16]:
					break
			cands.append(aux.map_idx(remaining,cand))
	elif gen == "LINCORR":
		lag = 5
		dep = np.zeros([N,N])
		for dat in train_data:
			#aux.print_mat(dat)
			dat = pp.normalize(dat,leave_zero=True)
			dep += aux.linear_dependence(dat,lag)
		dep = (dep.T + dep)*0.5
		dep = aux.normalize_corr_mat(dep)
		print("Correlation matrix: ")
		aux.print_mat(dep)

		subgroup_length = args.settings["subgroup_length"]
		cands = [aux.greedy_random_cand(dep,subgroup_length) for i in range(num)]
		cands = [aux.map_idx(remaining,cand) for cand in cands]

	return cands

## Training

# returns models = [obj]
def train(data,subgroup,args):
	#print(np.shape(data[0]))
	N = aux.num_features(data[0])

	train_type = args.model
	if train_type == "TRIVIAL":
		mod = Models.Trivial()
		mod.subgroup = subgroup
		mods = [mod]
	elif train_type == "VARMA":
		p = args.settings["VARMA_p"]
		q = args.settings["VARMA_q"]
		re_series = args.settings["re_series"]
		rw_series = args.settings["rw_series"]
		mods = [aux.train_varma(data,subgroup,p,q,re_series,rw_series)]
	elif train_type == "ARMA_PARALLEL":
		p = args.settings["VARMA_p"]
		q = args.settings["ARMA_q"]
		re_series = args.settings["re_series"]
		rw_series = args.settings["rw_series"]
		mods = [aux.train_varma([dat[:,i] for dat in data],[subgroup[i]],p,q,re_series,rw_series) for i in range(N)]

	return mods

def subgroup_learn(train_data,subgroup,args):
	#print(type(subgroup))
	if subgroup:
		mods = train([dat[:,subgroup] for dat in train_data],subgroup,args)
		return mods
	else:
		return []

def baseline_remain(train_data,sub_col,baseline,args):
	args = copy.copy(args)
	args.model = baseline

	#N = num_features(train_data[0])
	#remains = remaining_features(N,models)
	remains = sub_col.get_remaining()

	return subgroup_learn(train_data,remains,args)

## Testing

# returns either: predicted series or predicted classes/scores
def test(train_data,test_data,models,args):
	test_type = args.test_type
	k = int(args.pred_k)
	split_method = args.split_method

	pred = []
	if test_type == "PREDICTION":
		# take prediction unit by unit - units are assumed to be independent
		for tr_dat,test_dat in zip(train_data,test_data):
			# reset all inner states first
			aux.reset_models(models)

			# set states of models
			if split_method == "TIMEWISE":
				aux.update_models(tr_dat,models)

			pred.append(aux.predict_data(test_dat,models,k))

	elif test_type == "CLASSIFICATION":
		pass # how to do this with subgroups? not my problem for now

	return pred

# outputs performance scores, plots
# this function should be split into two: one evaluate() and one present()
def evaluate(pred,gt,evaluate_on,args):
	rmses = []
	for pred_mat,gt_mat in zip(pred,gt):
		#print(pred_mat)
		gt_mat = gt_mat[:,evaluate_on]
		pred_mat = pred_mat[:,evaluate_on]

		gt_mat,gt_mean,gt_std = pp.normalize(gt_mat,return_mean_std=True)
		pred_mat = pp.normalize_ref(pred_mat,gt_mean,gt_std)
		diff = pred_mat-gt_mat 
		fro = np.linalg.norm(diff)
		rms = fro/np.sqrt(np.size(pred_mat))
		rmses.append(rms)
		print("RMS norm of error: {0:.3f}".format(rms))
		
		if args.plot:
			for i,feature in enumerate(evaluate_on):
				plt.figure()
				plt.title(args.explanations[feature])
				plt.plot(pred_mat[:,i],'b')
				plt.plot(gt_mat[:,i],'r')
				plt.legend(["Predicted", "Ground Truth"])
				plt.xlabel("Time")
				plt.ylabel("Value")
			plt.show()
		
	rmses = np.array(rmses)
	print("Avg: {0:.3f}, Min: {1:.3f}, Max: {2:.3f}".format(np.mean(rmses),np.min(rmses),np.max(rmses)))

	return 0

def subgroup_score(train_data,test_data,sub_col,args,labels=[]):
	pred_k = int(args.pred_k)
	
	pred = test(train_data,[test_dat[:-pred_k] for test_dat in test_data],sub_col.candidates + sub_col.remaining,args)

	test_type = args.test_type.upper()
	if test_type == "PREDICTION":
		gt = [test_dat[pred_k:] for test_dat in test_data]
		N = aux.num_features(test_data[0])
		if args.subgroup_only_eval or not sub_col.remaining:
			evaluate_on = sub_col.get_covered()
			mods = sub_col.candidates
		else:
			evaluate_on = list(range(N))
			mods = sub_col.candidates + sub_col.remaining

		print("Evaluating on: ")
		for mod in mods:
			print(str(mod.subgroup) + ": " +type(mod).__name__)

		print(evaluate_on)
	elif test_type == "CLASSIFICATION":
		gt = labels
	else:
		print("Unknown test!")

	score = evaluate(pred,gt,evaluate_on,args)

	return score 

# alters models depending on what has become known with the new mod
def subgroup_select(mod,models,args):
	return False

## Main
def settings(args):
	num_series = 10

	settings = {"min_subgroup_length": 3, "max_subgroup_length": 6, "subgroup_length": 17, # general
				"VARMA_p": 2, "VARMA_q": 0, "ARMA_q": 2, # VARMA orders
				"re_series": np.logspace(-1,-6,num_series), "rw_series": 500*np.logspace(0,-1,num_series), # VARMA training
				"num_timepoints": 1000, "num_samples": 50, "case": "case4" # VARMA sim
				}

	args.settings = settings

def subgroup(data,args):
	train_data,test_data = pp.split(data,args.split_method,train_share=0.6)
	print(len(train_data))

	N = aux.num_features(data[0])
	sub_col = aux.Subgroup_collection(N,[])
	#for i in range(1):
	num = 20
	cands = candidate_generate(train_data,sub_col.get_remaining(),num,args)
	#for cand in cands:
	#	print(cand)

	'''
	legends = []
	print(args.explanations)
	for feature in [1,2,5]:
		plt.plot(data[0][:,feature])
		legends.append(args.explanations[feature])
	plt.legend(legends)
	plt.show()
	'''

	for cand in cands:
		print(cand)
		mods = subgroup_learn(train_data,cand,args)
		#for mod in mods:
		#	print(mod.q)
		sub_col.add(mods,"CANDIDATES")
		subgroup_score(train_data,test_data,sub_col,args)
		#if not subgroup_select(mods,sub_col,args):
		#	sub_col.remove_mods(mods)
		
		#rem = baseline_remain(train_data,sub_col,"ARMA_PARALLEL",args)
		#sub_col.add(rem,"REMAINING")

		#subgroup_score(train_data,test_data,sub_col,args)

		sub_col.reset()
	

def main(args):
	settings(args)
	data = preprocess(args)

	# subgroup learning
	subgroup(data,args)

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
	parser.add_argument('-s','--split_method',dest = 'split_method',default="UNITWISE",help='How to split data into training and test? (TIMEWISE,UNITWISE)')
	parser.add_argument('-t','--test_type',dest = 'test_type',default="PREDICTION",help='Type of learning')
	parser.add_argument('-g','--generation',dest = 'generation',default="TRIVIAL",help='Type of candidate generation')
	parser.add_argument('--subgroup_only_eval',dest = 'subgroup_only_eval',default=False,action="store_true",help='Only compare performance of subgroups')
	parser.add_argument('--plot',dest = 'plot',default=False,action="store_true",help='Show prediction plots')
	
	args = parser.parse_args()
	main(args)
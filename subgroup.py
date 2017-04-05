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
def fetch(args):
	dataset = args.dataset.upper()
	names = ""

	if dataset in ["TURBOFAN","MILL","IGBT"]:
		data,explanations = nasa.main(args)
	elif dataset == "BACKBLAZE":
		data,explanations,names = backblaze.main(args)
	#elif dataset == "ARMA_SIM":
	#	data = sim.arma_sim(np.array([1]),np.array([1,0.5,-0.2]),1000,num=5)
	elif dataset == "VARMA_SIM":
		if args.filename:
			data = sim.read(args.filename,args.elemsep,args.linesep)
			data = [pp.normalize(dat) for dat in data]
		else:
			num_timepoints = args.settings["num_timepoints"]
			num_samples = args.settings["num_samples"]
			case = args.settings["case"]
			data = [sim.mixed_varma(num_timepoints,case) for i in range(num_samples)]
			data = [pp.normalize(dat) for dat in data]
			sim.write(data,"VARMA",args)

		N = aux.num_features(data[0])
		explanations = ["feature {0:d}".format(i) for i in range(N)]
	else:
		print("No matching dataset!")

	args.explanations = explanations
	args.names = names

	#for dat,name in zip(data,names):
	#	print(name)
	#	aux.whiteness_test([dat],explanations)
	#aux.whiteness_test(data,explanations)

	return data

## Candidate Generation

def candidate_generate(train_data,remaining,num,args):
	N = len(remaining)

	gen = args.generation.upper()
	if gen == "TRIVIAL":
		cand = list(range(N))
		cands = [aux.map_idx(remaining,cand)]*num # subgroups won't be edited, only read
	elif gen == "CUSTOM":
		#cands = [[8,13,15,18,19,21]] # turbofan 2
		#cand = [2,6,9,10,11]
		cand = [0,1,2]
		ext = [] #[elem+12 for elem in cand]
		cands = [cand + ext]
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
		lag = args.settings["lincorr_lag"]
		dep = np.zeros([N,N])
		for dat in train_data:
			#aux.print_mat(dat)
			dat = pp.normalize(dat,leave_zero=True)
			dep += aux.linear_dependence(dat,lag)
		dep = (dep.T + dep)*0.5
		dep = aux.normalize_corr_mat(dep)
		dep = np.abs(dep)
		print("Correlation matrix: ")
		aux.print_mat(dep)

		subgroup_length = args.settings["subgroup_length"]
		cands = [aux.greedy_random_cand(dep,subgroup_length) for i in range(num)]
		cands = [aux.map_idx(remaining,cand) for cand in cands]

	return cands

## Training

# returns models = [obj]
def train(data,subgroup,train_type,args):
	#print(np.shape(data[0]))
	N = aux.num_features(data[0])

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
	elif train_type == "ESN":
		A_arch = args.settings["A_architecture"]
		B_arch = args.settings["B_architecture"]
		C_arch = args.settings["C_architecture"]
		f_arch = args.settings["f_architecture"]
		size_nodes = args.settings["ESN_size_state"]
		size_out = args.settings["ESN_size_out"]
		re_series = args.settings["re_series"]
		rw_series = args.settings["rw_series"]
		burn_in = args.settings["ESN_burn_in"]
		batch_train = args.settings["ESN_batch_train"]
		tikho = args.settings["ESN_tikhonov_const"]
		style = args.test_type

		mods = [aux.train_esn(data,subgroup,style,[N,size_nodes,size_out,N],[A_arch,B_arch,C_arch,f_arch],re_series,rw_series,burn_in,batch_train,tikho)]

		#mods[0].print_esn()

	elif train_type == "SVM":
		mod = Models.SVM_TS(subgroup,args.settings["pos_w"],args.settings["style"])
		for X,y in aux.impending_failure(data,args.test_names,args.dataset,args.settings["failure_horizon"],mod.style):
			mod.update(X,y)

		mod.train()

		mods = [mod]

	return mods

def subgroup_learn(train_data,subgroup,train_type,args):
	#print(type(subgroup))
	if subgroup:
		mods = train([dat[:,subgroup] for dat in train_data],subgroup,train_type,args)
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
	labels = []
	if test_type == "PREDICTION":
		# take prediction unit by unit - units are assumed to be independent
		
		# reset all inner states first
		aux.reset_models(models)
		if split_method == "TIMEWISE":	
			for tr_dat,test_dat in zip(train_data,test_data):
				# set states of models
				aux.update_models(tr_dat,models)

				pred.append(aux.predict_data(test_dat,models,k))
		elif split_method == "UNITWISE":
			for test_dat in test_data:
				pred.append(aux.predict_data(test_dat,models,k))

	elif test_type == "CLASSIFICATION":
		if args.model == "SVM":
			mod = models[0] # assume that there is just one at first
			#print(len(test_data))
			#print(len(args.test_names))
			for X,y in aux.impending_failure(test_data,args.train_names,args.dataset,args.settings["failure_horizon"],mod.style):
				X = X[:,mod.subgroup]
				pred.append(mod.predict(X))
				labels.append(y)

			#pred = np.concatenate(pred)
			#labels = np.concatenate(labels)

	return pred,labels

# outputs performance scores, plots
# this function should be split into two: one evaluate() and one present()
def evaluate(pred,gt,evaluate_on,args):
	test_type = args.test_type.upper()
	rmses = []
	j = 0

	if test_type == "PREDICTION":
		for pred_mat,gt_mat in zip(pred,gt):
			#print(pred_mat)
			gt_mat = gt_mat[:,evaluate_on]
			pred_mat = pred_mat[:,evaluate_on]

			#gt_mat,gt_mean,gt_std = pp.normalize(gt_mat,return_mean_std=True,leave_zero=True)
			#pred_mat = pp.normalize_ref(pred_mat,gt_mean,gt_std)
			diff = pred_mat-gt_mat 
			fro = np.linalg.norm(diff)
			rms = fro/np.sqrt(np.size(pred_mat))
			rmses.append(rms)
			
			if args.test_names:
				print("{0:s}, RMS norm of error: {1:.3f}".format(args.test_names[j],rms))
			else:
				print("RMS norm of error: {0:.3f}".format(rms))
			
			if args.plot:
				for i,feature in enumerate(evaluate_on):
					plt.figure()
					plt.title(args.explanations[feature]+". {0:d}-step prediction.".format(int(args.pred_k)))
					plt.plot(pred_mat[:,i],'b')
					plt.plot(gt_mat[:,i],'r')
					plt.plot(diff[:,i],'g')
					plt.legend(["Predicted", "Ground Truth","Residual"])
					plt.xlabel("Time")
					plt.ylabel("Value")
				plt.show()

			j+=1

	elif test_type == "CLASSIFICATION":
		rmses = [1]
		aux.classification_plot(pred,gt,args.settings["style"],args.settings["failure_horizon"])
		
	rmses = np.array(rmses)
	print("Avg: {0:.3f}, Min: {1:.3f}, Max: {2:.3f}".format(np.mean(rmses),np.min(rmses),np.max(rmses)))

	return 0

def subgroup_score(train_data,test_data,sub_col,args,labels=[]):
	pred_k = int(args.pred_k)
	
	test_type = args.test_type.upper()
	if test_type == "PREDICTION":
		pred,__ = test(train_data,[test_dat[:-pred_k] for test_dat in test_data],sub_col.candidates + sub_col.remaining,args)
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
		pred,gt = test(train_data,test_data,sub_col.candidates + sub_col.remaining,args)
		evaluate_on = sub_col.candidates[0]
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

	settings = {"min_subgroup_length": 3, "max_subgroup_length": 6, "subgroup_length": 3, # general
				"lincorr_lag": 5, # candidate generation
				"VARMA_p": 2, "VARMA_q": 0, "ARMA_q": 2, # VARMA orders
				"re_series": np.logspace(-1,-6,num_series), "rw_series": 500*np.logspace(0,-1,num_series), # VARMA training
				"num_timepoints": 1000, "num_samples": 50, "case": "case1", # VARMA sim
				"train_share": 0.6, "test_share": 0.2, # splitting
				"failure_horizon": 10, "pos_w": 5, "style": "MLP", # SVM 
				"A_architecture": "SCR", "B_architecture": "SECTIONS", "C_architecture": "SELECTED", "f_architecture": "TANH", # ESN
				"ESN_size_state": 500, "ESN_size_out": 30, # ESN
				"ESN_burn_in": 10,"ESN_batch_train" : True,"ESN_tikhonov_const": 3  # ESN training
				}

	args.settings = settings

def subgroup(data,args):
	train_data,test_data,train_names,test_names = pp.split(data,args.split_method,train_share=args.settings["train_share"],test_share=args.settings["test_share"],names=args.names,return_names=True)
	args.train_names = train_names
	args.test_names = test_names
	print(len(train_data))
	
	#print(data[0])
	#print(data[0].shape)

	N = aux.num_features(data[0])
	sub_col = aux.Subgroup_collection(N,[])
	#for i in range(1):
	num = 1
	cands = candidate_generate(train_data,sub_col.get_remaining(),num,args)
	#for cand in cands:
	#	print(cand)

	if 0:
		legends = []
		print(args.explanations)
		for feature in [1,2,5]:
			plt.plot(data[0][:,feature])
			legends.append(args.explanations[feature])
		plt.legend(legends)
		plt.show()

	for cand in cands:
		print(cand)
		for model in [args.model]:
			print(model)
			mods = subgroup_learn(train_data,cand,model,args)
			#print(mods[0].Cs)
			#for mod in mods:
			#	print(mod.q)
			sub_col.add(mods,"CANDIDATES")
			subgroup_score(train_data,test_data,sub_col,args)

			sub_col.reset()
			#if not subgroup_select(mods,sub_col,args):
			#	sub_col.remove_mods(mods)
			
			#rem = baseline_remain(train_data,sub_col,"ARMA_PARALLEL",args)
			#sub_col.add(rem,"REMAINING")

			#subgroup_score(train_data,test_data,sub_col,args)
	
	
def main(args):
	settings(args)
	data = fetch(args)

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
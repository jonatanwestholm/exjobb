# subgroup.py

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

import settings
import subgroup_auxilliary as aux
import preprocessing as pp
import sim
import models as Models
import nasa
import backblaze
import occupancy
import dodgers

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
	gt = []

	if dataset in ["TURBOFAN","MILL","IGBT"]:
		data,gt,explanations = nasa.main(args)
	elif dataset == "BACKBLAZE":
		data,gt,explanations,names = backblaze.main(args)
	elif dataset == "OCCUPANCY":
		data,gt,explanations,names = occupancy.main(args)
	elif dataset == "DODGERS":
		data,gt,explanations,names = dodgers.main(args)
	#elif dataset == "ARMA_SIM":
	#	data = sim.arma_sim(np.array([1]),np.array([1,0.5,-0.2]),1000,num=5)
	elif dataset == "VARMA_SIM":
		if args.filename:
			data,gt = sim.read(args.filename,args.elemsep,args.linesep)
			data = [pp.normalize(dat) for dat in data]
		else:
			num_timepoints = args.settings["num_timepoints"]
			num_samples = args.settings["num_samples"]
			case = args.settings["case"]
			data = [sim.mixed_varma(num_timepoints,case) for i in range(num_samples)]
			data = [pp.normalize(dat) for dat in data]
			sim.write(data,gt,"VARMA",args)

		N = aux.num_features(data[0])
		explanations = ["feature {0:d}".format(i) for i in range(N)]
	elif dataset == "ESN_SIM":
		if args.filename:
			data,gt = sim.read(args.filename,args.elemsep,args.linesep)
			#data = [pp.normalize(dat) for dat in data]
		else:
			num_timepoints = args.settings["num_timepoints"]
			num_samples = args.settings["num_samples"]
			case = args.settings["ESN_sim_case"]
			data = []
			gt = []
			for i in range(num_samples):
				dat,gt_inst = sim.esn_sim(num_timepoints,case)
				data.append(dat)
				gt.append(gt_inst)

			#data = [pp.normalize(dat) for dat in data]
			sim.write(data,gt,"ESN",args)

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

	return data,gt

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
def train(train_data,train_gt,subgroup,train_type,args):
	#print(np.shape(data[0]))
	test_type = args.test_type
	N = aux.num_features(train_data[0])

	if train_type == "TRIVIAL":
		mod = Models.Trivial()
		mod.subgroup = subgroup
		mods = [mod]
	elif train_type == "VARMA":
		p = args.settings["VARMA_p"]
		q = args.settings["VARMA_q"]
		re_series = args.settings["re_series"]
		rw_series = args.settings["rw_series"]
		mods = [aux.train_varma(train_data,subgroup,p,q,re_series,rw_series)]
	elif train_type == "ARMA_PARALLEL":
		p = args.settings["VARMA_p"]
		q = args.settings["ARMA_q"]
		re_series = args.settings["re_series"]
		rw_series = args.settings["rw_series"]
		mods = [aux.train_varma([dat[:,i] for dat in train_data],[subgroup[i]],p,q,re_series,rw_series) for i in range(N)]
	elif train_type == "ESN":
		spec = args.settings["ESN_spec"]
		mixing = args.settings["ESN_mixing"]
		size_out = args.settings["ESN_size_out"]
		burn_in = args.settings["ESN_burn_in"]
		#batch_train = args.settings["ESN_batch_train"]
		tikho = args.settings["ESN_tikhonov_const"]
		purpose = args.test_type
		pos_w = args.settings["pos_w"]
		#sig_limit = args.settings["ESN_sig_limit"]
		selection = args.settings["ESN_feature_selection"]
		classifier = args.settings["ESN_classifier"]
		explanations = args.explanations

		if test_type == "PREDICTION":
			L = N
		else:
			L = 1
		orders = [N,size_out,L]

		mod = Models.ESN(purpose,orders,spec,mixing,pos_w,selection,classifier,explanations)
		mod.subgroup = subgroup

		if test_type == "PREDICTION":
			train_gt = [None]*len(train_data)

		for x,y in zip(train_data,train_gt):
			mod.charge(x,y)

		X_res = mod.train(tikho)

		mods = [mod]

	elif train_type == "SVM":
		mod = Models.SVM_TS(subgroup,args.settings["pos_w"],test_type)
		for x,y in zip(train_data,train_gt):
			mod.charge(x,y)
		mod.train()
		mods = [mod]
	elif train_type == "MLP":
		mod = Models.MLP_TS(subgroup,test_type)
		for x,y in zip(train_data,train_gt):
			mod.charge(x,y)
		mod.train()
		mods = [mod]

	return mods

def subgroup_learn(train_data,train_gt,subgroup,train_type,args):
	#print(type(subgroup))
	if subgroup:
		mods = train([dat[:,subgroup] for dat in train_data],train_gt,subgroup,train_type,args)
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
		
		if split_method == "TIMEWISE":	
			for tr_dat,test_dat in zip(train_data,test_data):
				# set states of models
				aux.reset_models(models)
				aux.update_models(tr_dat,models)

				pred.append(aux.predict_data(test_dat,models,k))
		elif split_method == "UNITWISE":
			for test_dat in test_data:
				aux.reset_models(models)
				pred.append(aux.predict_data(test_dat,models,k))

	elif test_type in ["CLASSIFICATION","REGRESSION"]:
		mod = models[0]
		for test_dat in test_data:
			mod.reset()
			pred.append(mod.predict(U=test_dat))

	return pred

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

	elif test_type in ["CLASSIFICATION","REGRESSION"]:
		G_all = np.array([[0]])
		GG_all = 0
		PG_all = 0
		PP_all = 0
		for P,G in zip(pred,gt):
			G_all = np.concatenate([G_all,G])
			GG = np.sum(G)
			PG = float(np.dot(P.T,G))
			PP = np.sum(P)
			GG_all += GG
			PG_all += PG
			PP_all += PP
			spec,prec,hm = aux.classification_stats(GG,PG,PP)
			if args.dataset in ["DODGERS"]:
				spec = aux.interval_hits(P,G)
				try:
					spec = spec[0]
				except IndexError:
					pass
				hm = 2/(1/spec + 1/prec)
			if args.test_names:
				test_name = args.test_names[j]
			else:
				test_name = "Fig {0:d}".format(j)
			triv = 2*np.mean(G)/(1+np.mean(G))
			print("Unit {0:s}, spec: {1:.3f} prec: {2:.3f}, hm: {3:.3f}, trivial: {4:.3f}".format(test_name,spec,prec,hm,triv))

			j += 1

		G_all = G_all[1:]
		triv = 2*np.mean(G_all)/(1+np.mean(G_all))
		spec,prec,hm = aux.classification_stats(GG_all,PG_all,PP_all)
		print("Total. spec: {0:.3f} prec: {1:.3f}, hm: {2:.3f}, trivial: {3:.3f}".format(spec,prec,hm,triv))
			
		if args.plot:
			aux.classification_plot(pred,gt,args.names)

	elif test_type == "REGRESSION":
		pass		

	return 0

def subgroup_score(train_data,test_data,gt,sub_col,args):
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
	elif test_type in ["CLASSIFICATION","REGRESSION"]:
		pred = test(train_data,test_data,sub_col.candidates + sub_col.remaining,args)
		evaluate_on = sub_col.candidates[0]
	else:
		print("Unknown test!")

	score = evaluate(pred,gt,evaluate_on,args)

	return score 

## Main

# example:
# spec = {"DIRECT": None,"VAR": {"p": 5}, "RODAN": {"N": 200}, "THRES": {"random_thres": True, "N": 20}}

def set_settings(args):
	if args.config:
		config = args.config
	else:
		config = args.dataset

	args.settings = settings.settings[config]

def subgroup(data,gt,args):
	train_data,train_gt,test_data,test_gt,train_names,test_names = pp.split(data,gt,
																			args.split_method,
																			train_share=args.settings["train_share"],
																			test_share=args.settings["test_share"],
																			names=args.names,
																			return_names=True)
	print(train_names)
	print(test_names)

	try: 
		if args.settings["self_test"]:
			test_data = train_data
			test_gt = train_gt
			test_names = train_names
	except KeyError:
		pass
	
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

	for cand in cands:
		print(cand)
		for model in [args.model]:
			print(model)
			mods = subgroup_learn(train_data,train_gt,cand,model,args)
			#print(mods[0].Cs)
			#for mod in mods:
			#	print(mod.q)
			sub_col.add(mods,"CANDIDATES")
			subgroup_score(train_data,test_data,test_gt,sub_col,args)

			sub_col.reset()
			#if not subgroup_select(mods,sub_col,args):
			#	sub_col.remove_mods(mods)
			
			#rem = baseline_remain(train_data,sub_col,"ARMA_PARALLEL",args)
			#sub_col.add(rem,"REMAINING")

			#subgroup_score(train_data,test_data,sub_col,args)
	
	
def main(args):
	set_settings(args)
	data,gt = fetch(args)

	# subgroup learning
	subgroup(data,gt,args)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--filename',dest = 'filename',default="",help='Name of input file')
	#parser.add_argument('-t','--target',dest = 'target',default="",help='Name of target directory for melting')
	parser.add_argument('-p','--pattern',dest = 'pattern',default="",help='Input file pattern (regex)')
	parser.add_argument('-a','--datatype',dest = 'datatype',default="SEQUENTIAL",help='Type of data (e.g. SEQUENTIAL,INSTANCE,LAYERED)')
	parser.add_argument('-d','--dataset',dest = 'dataset',default="",help='Pick dataset (e.g. TURBOFAN,IGBT,BEARING,MILL)')
	parser.add_argument('--config',dest = 'config',default="",help='Config to use for models')
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
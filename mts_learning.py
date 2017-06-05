import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

import settings
import mts_learning_auxilliary as aux
import preprocessing as pp
import sim
import models as Models
import nasa
import backblaze
import occupancy
import dodgers
import eye

## Preprocessing

# returns data = [np.array()]
# preprocessing is dataset specific so it's handled in separate scripts
def fetch(args,dataset="",number=0):
	if not dataset:
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
	elif dataset == "EYE":
		data,gt,explanations,names = eye.main(args)
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
		names = ["unit {0:d}".format(i) for i in range(num_samples)]
	elif dataset == "PROBE":
		data,gt = sim.esn_sim(args.settings["num_timepoints"],"PROBE",number)
			
	else:
		print("No matching dataset!")

	if "PROBE" not in dataset:
		args.explanations = explanations
		args.names = names

	#for dat,name in zip(data,names):
	#	print(name)
	#	aux.whiteness_test([dat],explanations)
	#aux.whiteness_test(data,explanations)

	return data,gt

## Training

# returns models = [obj]
def train(train_data,train_gt,train_type,args):
	#print(np.shape(data[0]))
	test_type = args.test_type
	N = aux.num_features(train_data[0])

	if train_type == "ESN":
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

		L = 1
		orders = [N,size_out,L]

		mod = Models.ESN(purpose,orders,spec,mixing,pos_w,selection,classifier,explanations)

		for x,y in zip(train_data,train_gt):
			mod.charge(x,y,burn_in=burn_in)

		X_res = mod.train(tikho)

	elif train_type == "SVM":
		mod = Models.SVM_TS(N,args.settings["pos_w"],test_type)
		for x,y in zip(train_data,train_gt):
			mod.charge(x,y)
		mod.train()
	elif train_type == "MLP":
		mod = Models.MLP_TS(N,test_type)
		for x,y in zip(train_data,train_gt):
			mod.charge(x,y)
		mod.train()

	return mod

## Testing

# returns either: predicted series or predicted classes/scores
def test(test_data,model,args):
	pred = []
	for test_dat in test_data:
		model.reset()
		pred.append(model.predict(U=test_dat))

	return pred

# outputs performance scores, plots
# this function should be split into two: one evaluate() and one present()
def evaluate(test_data,pred,gt,args):
	test_type = args.test_type.upper()
	rmses = []
	j = 0

	G_all = np.array([[0]])
	GG_all = 0
	PG_all = 0
	PP_all = 0
	E_all = 0
	T_all = 0
	hits_total = 0
	for P,G in zip(pred,gt):
		#E = np.sum(np.abs(P-G.reshape([len(G),])))
		P = P.reshape([len(P),1])
		G = G.reshape([len(G),1])
		E = np.sum(np.abs(P-G))
		E_all += E 
		T = np.prod(G.shape)
		T_all += T
		acc = 1-E/T

		G_all = np.concatenate([G_all,G])
		GG = np.sum(G)
		PG = float(np.dot(P.T,G))
		PP = np.sum(P)
		GG_all += GG
		PG_all += PG
		PP_all += PP
		spec,prec,hm = aux.classification_stats(GG,PG,PP)
		if args.dataset in ["DODGERS","BACKBLAZE"]:
			spec = aux.interval_hits(P,G)
			hits_total += spec
			try:
				spec = spec[0]
			except IndexError:
				pass
			if spec:
				hm = 2/(1/spec + 1/prec)
		if args.test_names:
			test_name = args.test_names[j]
		else:
			test_name = "Fig {0:d}".format(j)
		triv = 2*np.mean(G)/(1+np.mean(G))
		print("Unit {0:s}, spec: {1:.3f} prec: {2:.3f}, acc: {3:.3f}, hm: {4:.3f}, trivial: {5:.3f}".format(test_name,spec,prec,acc,hm,triv))

		j += 1

	G_all = G_all[1:]
	triv = 2*np.mean(G_all)/(1+np.mean(G_all))
	spec,prec,hm = aux.classification_stats(GG_all,PG_all,PP_all)
	if args.dataset in ["BACKBLAZE"]:
		spec = hits_total/len(pred)
		spec = spec[0]
		hm = 2/(1/spec + 1/prec)
	acc = 1 - E_all/T_all
	print("Total. spec: {0:.3f} prec: {1:.3f}, acc: {2:.3f}, hm: {3:.3f}, trivial: {4:.3f}".format(spec,prec,acc,hm,triv))
		
	if 1: #args.dataset in ["BACKBLAZE"]:
		aux.burn_down_graph(pred,[1,2,3,4,5])

	if args.plot:
		aux.classification_plot(test_data,pred,gt,args.test_names,args.dataset,args.model)

## Main

# example:
# spec = {"DIRECT": None,"VAR": {"p": 5}, "RODAN": {"N": 200}, "THRES": {"random_thres": True, "N": 20}}

def set_settings(args):
	if args.config:
		config = args.config
	else:
		config = args.dataset

	args.settings = settings.settings[config]

def test_model(model,data,gt,args):
	print("Testing")
	pred = test(data,model,args)
	print("Evaluating")
	score = evaluate(data,pred,gt,args)

def main(args):
	set_settings(args)
	data,gt = fetch(args)
	train_data,train_gt,test_data,test_gt,train_names,test_names = pp.split(data,gt,
																			args.split_method,
																			train_share=args.settings["train_share"],
																			test_share=args.settings["test_share"],
																			names=args.names,
																			return_names=True)
	#print(train_names)
	print(len(train_data))
	#print(test_names)
	print(len(test_data))

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

	#probe_data,probe_gt = fetch(args,"PROBE",data[0].shape[1])
	#print(len(probe_data))
	#print(probe_data[0])

	print("Training")
	model = train(train_data,train_gt,args.model,args)
	test_model(model,test_data,test_gt,args)

	#args.test_names = args.explanations
	#test_model(model,probe_data,probe_gt,args)




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
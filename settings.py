#settings.py

backblaze = {#"min_subgroup_length": 3, "max_subgroup_length": 6, "subgroup_length": 3, # general
			#"lincorr_lag": 5, # candidate generation
			#"VARMA_p": 2, "VARMA_q": 0, "ARMA_q": 2, # VARMA orders
			#"re_series": np.logspace(-1,-6,num_series), "rw_series": 500*np.logspace(0,-1,num_series), # VARMA training
			"num_timepoints": 300, #"num_samples": 10, "case": "case1", # VARMA sim
			"train_share": 0.3, "test_share": 0.3, # splitting
			"failure_horizon": 30, 
			"pos_w": 5, # classification and regression
			#"A_architecture": "DLR", "B_architecture": "SECTIONS", "C_architecture": "SELECTED", "f_architecture": "TANH", # ESN
			#"ESN_size_state": 500, 998
			"ESN_spec": [#("RODAN", {"N": 500,"r":0.9,"v":0}),
						("RODAN",{"N":500,"v":0,"r":0.7}),
						("EXPDELAY", {"N": 13, "order": 6, "direct_input": False}),
						#("VAR", {"p": 20}),
						#("THRES", {"N": 200,"random_thres":True,"direct_input":True}),
						#("TRIGGER", {"N": 300,"random_thres": True,"direct_input":False}),
						#("LEAKY", {"N": 200, "r": 0.8,"v":1}),
						#("HEIGHTSENS", {"N": 200, "random_thres": True}),
						("DIRECT",{}),
						],
			"ESN_size_out": 20, # ESN
			"ESN_burn_in": 60,"ESN_batch_train" : True,"ESN_tikhonov_const": 1,  # ESN training
			"ESN_mixing": [("DIRECT","EXPDELAY",13,False,0.001),("EXPDELAY","RODAN",1000,True,500),("EXPDELAY","TRIGGER",100),("RODAN","TRIGGER",100),
							("RODAN","RODAN",2000,True,0.5)],
			#"ESN_mixing": [("RODAN","RODAN",200),("TRIGGER","RODAN",200),("RODAN","TRIGGER",200),
			#			   ("THRES","VAR",1), ("VAR","TRIGGER",1), ("LEAKY","TRIGGER",20), ("THRES","LEAKY",50), ("LEAKY","RODAN",100),
			#			   ("HEIGHTSENS","HEIGHTSENS",1),("HEIGHTSENS","LEAKY",200),("RODAN","HEIGHTSENS",200),
			#			   ("VAR","RODAN",10)],
			#"ESN_rebuild_types": ["THRES","TRIGGER"], "ESN_rebuild_iterations": 1, "ESN_impact_limit": 1e-2,
			"ESN_feature_selection": "SVD_SEP",
			"ESN_classifier": "LINEAR" #, "ESN_sig_limit": 1.1
			}

dodgers = {#"train_share": 0.4, "test_share": 0.3, # splitting
		   #"train_share": 0.7, "test_share": 0.3, # splitting
		   "train_share": 1, "test_share": 0, "self_test": True, # splitting
			"pos_w": 3, # classification and regression
			"ESN_spec": [#("RODAN", {"N": 200,"v":0}),
						("RODAN",{"N":200,"v":0.5,"r":0.5}),
						#("EXPDELAY", {"N": 50, "order": 3, "direct_input": False}),
						("VAR", {"p": 10}),
						("THRES", {"N": 50,"random_thres":True,"direct_input":False}),
						#("TRIGGER", {"N": 500,"random_thres": True,"direct_input":True}),
						#("LEAKY", {"N": 200, "r": 0.9,"v":0.3}),
						#("HEIGHTSENS", {"N": 200, "random_thres": True}),
						#("DIRECT",{}),
						],
			"ESN_size_out": 7, # ESN
			"ESN_burn_in": 10,"ESN_batch_train" : True,"ESN_tikhonov_const": 3,  # ESN training
			"ESN_mixing": [("RODAN","RODAN",50),("LEAKY","THRES",100),("RODAN","THRES",50),("RODAN","LEAKY",200),("VAR","THRES",10),("VAR","RODAN",10),
							("RODAN","EXPDELAY",50,False,1),("EXPDELAY","RODAN",150,False,1)],
			#"ESN_rebuild_types": ["THRES","TRIGGER"], "ESN_rebuild_iterations": 1, "ESN_impact_limit": 1e-2,
			"ESN_feature_selection": "SVD",
			"ESN_classifier": "MLP"
			}

occupancy = {#"train_share": 0.4, "test_share": 0.4, "self_test": True,# splitting
			 "train_share": 0.4, "test_share": 0.6, "self_test": False,# splitting
			"pos_w": 1, # classification and regression
			"ESN_spec": [#("RODAN", {"N": 500,"v":0}),
						("RODAN",{"N": 200,"v":0.5,"r":0.9}),
						("VAR", {"p": 10}),
						("THRES", {"N": 300,"random_thres":True,"direct_input":True}),
						#("TRIGGER", {"N": 500,"random_thres": True,"direct_input":True}),
						#("EXPDELAY", {"N": 5, "order": 5, "direct_input": False}),
						("LEAKY", {"N": 100, "r": 0.7,"v":1}),
						#("LEAKY", {"N":100,"r":0.9,"v":0}),
						#("HEIGHTSENS", {"N": 200, "random_thres": True}),
						("DIRECT",{}),
						],
			"ESN_size_out": 10, # ESN
			"ESN_burn_in": 10,"ESN_batch_train" : True,"ESN_tikhonov_const": 1,  # ESN training
			#"ESN_mixing": [("DIRECT","EXPDELAY",5,False,1),("EXPDELAY","RODAN",100,True,1),("RODAN","RODAN",200,False,0.1)],
			"ESN_mixing": [('RODAN', 'RODAN', 200), ('LEAKY', 'RODAN', 50), ('THRES', 'LEAKY', 100), ('VAR', 'RODAN', 50), ('RODAN', 'THRES', 200)],
			#"ESN_rebuild_types": ["THRES","TRIGGER"], "ESN_rebuild_iterations": 1, "ESN_impact_limit": 1e-2,
			"ESN_feature_selection": "SVD_SEP",
			"ESN_classifier": "LINEAR" #, "ESN_sig_limit": 0.1,
			}

eye = {"train_share": 0.7, "test_share": 0.3, "self_test": False,# splitting
	   #"train_share": 0.4, "test_share": 0.6, "self_test": False,# splitting
	   		"num_timepoints": 200,
			"pos_w": 1, # classification and regression
			"ESN_spec": [("RODAN", {"N": 500,"v":0,"r":0.9}),
						#("RODAN",{"N": 1000,"v":0.5,"r":0.9}),
						("EXPDELAY", {"N": 14, "order": 10, "direct_input": False}),
						#("VAR", {"p": 10}),
						#("THRES", {"N": 300,"random_thres":True,"direct_input":True}),
						#("TRIGGER", {"N": 500,"random_thres": True,"direct_input":True}),
						#("LEAKY", {"N": 100, "r": 0.7,"v":1}),
						#("LEAKY", {"N":100,"r":0.9,"v":0}),
						#("HEIGHTSENS", {"N": 200, "random_thres": True}),
						("DIRECT",{}),
						],
			"ESN_size_out": 20, # ESN
			"ESN_burn_in": 10,"ESN_batch_train" : True,"ESN_tikhonov_const": 10,  # ESN training
			"ESN_mixing": [("DIRECT","EXPDELAY",14,False,0.001),("EXPDELAY","RODAN",200,True,500),("RODAN","EXPDELAY",0),("EXPDELAY","RODAN",0),("RODAN","RODAN",200)],
			#"ESN_rebuild_types": ["THRES","TRIGGER"], "ESN_rebuild_iterations": 1, "ESN_impact_limit": 1e-2,
			"ESN_feature_selection": "SVD_SEP",
			"ESN_classifier": "SVM" #, "ESN_sig_limit": 0.1,
			}

esn_sim = {#"train_share": 0.4, "test_share": 0.4, "self_test": True,# splitting
			"train_share": 0.5, "test_share": 0.5, "self_test": False,# splitting
			"num_samples": 2, #Simulation of data
			"num_timepoints": 200,
			"ESN_sim_case": "step",
			"pos_w": 10, # classification and regression
			"ESN_spec": [("DIRECT",{}),
						#("CHAIN", {"order": 4, "v": 0.5, "r": 0.5}),
						#("RODAN", {"N": 100,"v":0,"r": 0.5}),
						#("RODAN",{"N": 200,"v":0.5,"r":0.5}),
						("EXPDELAY", {"N": 1, "order": 5, "direct_input": False}),
						#("VAR", {"p": 5}),
						#("THRES", {"N": 50,"random_thres":True,"direct_input":True}),
						#("TRIGGER", {"N": 500,"random_thres": True,"direct_input":True}),
						#("LEAKY", {"N": 100, "r": 0.7,"v":1}),
						#("LEAKY", {"N":100,"r":0.9,"v":0}),
						#("HEIGHTSENS", {"N": 200, "random_thres": True}),
						],
			"ESN_size_out": 2, # ESN
			"ESN_burn_in": 0,"ESN_batch_train" : True,"ESN_tikhonov_const": 1,  # ESN training
			"ESN_mixing": [("DIRECT","EXPDELAY",5,True),("RODAN","RODAN",0,False),("RODAN","EXPDELAY",50,False,1),("EXPDELAY","RODAN",50,False,1)],
			#"ESN_rebuild_types": ["THRES","TRIGGER"], "ESN_rebuild_iterations": 1, "ESN_impact_limit": 1e-2,
			"ESN_feature_selection": "SVD",
			"ESN_classifier": "LINEAR" #, "ESN_sig_limit": 0.1,
			}

settings = {"DODGERS": dodgers, "OCCUPANCY": occupancy, "BACKBLAZE": backblaze,"EYE": eye,"ESN_SIM": esn_sim}

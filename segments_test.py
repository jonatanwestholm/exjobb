#segments_test.py

import argparse
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier,MLPRegressor

import preprocessing as pp
import models
import sim
import eye
import backblaze

def get_segments(x,length,strafe=1):
	return [x[i:i+length] for i in range(0,len(x)-length,strafe)]

def plot_cluster(X,C,cluster):
	plt.figure()
	plt.title(cluster)
	for idx,c in enumerate(C):
		if c == cluster:
			plt.plot(X[idx,:])

def train_perceptron(X,C,cluster):
	perc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3), random_state=1)
	Y = [c == cluster for c in C]

	perc.fit(X,Y)

	return perc

def print_mat(mat):
	print()
	for row in mat:
		print("".join(["{0:.3f}".format(elem).rjust(10,' ') for elem in row]))

def naive_downsample(e,rate=2):
	return e[::rate,:]

def naive_upsample(e,rate=2):
	N = len(e)
	e = np.concatenate([e]*rate,axis=1)
	return e.reshape([N*rate,1])

def normalize_corr_mat(dep):
	#autoprecisions = forgiving_inv(np.diag(dep))
	autocorr = np.sqrt(np.diag(1/np.diag(dep)))
	#print_mat(autocorr)
	return np.dot(autocorr,np.dot(dep,autocorr))

def main(args):
	args.test_type = "CLASSIFICATION"
	args.settings = {}
	args.settings["failure_horizon"] = 30 #just a dummy

	num = 10
	N = 100
	seg_length = 6
	strafe = 1
	sample_rate = 4
	#e,__ = sim.esn_sim(N,"denoising_test")
	#e,_,_,_ = eye.main(args)
	e,_,_,_ = backblaze.main(args)
	e = np.concatenate(e,axis=0)

	e = e[:,0]
	e = e.reshape([len(e),1])
	print(e.shape)

	#print("Number of NaNs: {0:d}".format(len(np.where(np.isnan(e))[0])))
	e_small = naive_downsample(e,sample_rate)

	#plt.plot(e)
	#plt.plot(naive_upsample(e_small,sample_rate))
	#plt.plot(e_small)
	#plt.show()

	E = get_segments(np.ravel(e_small),seg_length,strafe)
	E = np.array(E).T
	E = pp.normalize(E,leave_zero=True).T

	#print(e)
	#print(np.array(E))

	mod = KMeans(n_clusters=num).fit(E)

	C = mod.predict(E)

	'''
	for i in range(num):
		plot_cluster(E,C,i)

	plt.show()
	'''

	split = int(0.5*E.shape[0])
	X = E[:split,:
]	C1 = C[:split]
	print(X.shape)
	print(C1.shape)

	perceptrons = [train_perceptron(X,C1,i) for i in range(num)]

	X = E[split:,:]
	C2 = C[split:]

	gen = [perc.predict_proba(X)[:,1] for perc in perceptrons]
	gen = np.array(gen).T


	dep = np.dot(gen.T,gen)
	dep = normalize_corr_mat(dep)

	print_mat(dep)

	plt.figure()
	plt.plot(e_small[split:]+num)
	plt.plot(C2)
	#plt.plot(naive_upsample(e_small,sample_rate)+num)
	for i,perc in zip(range(num),perceptrons):
		Y = perc.predict_proba(X)[:,1]+i
		Y = Y.reshape([len(Y),1])
		#Y = naive_upsample(Y,sample_rate)
		plt.plot(list(range(0,X.shape[0]*straf,sterafe)),Y)

	legends = ["Signal"]+["Cluster {0:d} match".format(i) for i in range(num)]
	plt.legend(legends)
	#plot_cluster(X,C,cluster)

	for i in range(num):
		plot_cluster(E,C,i)

	plt.show()
	


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

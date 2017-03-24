#svmtest.py

import numpy as np 
import sklearn.svm as svm
import matplotlib.pyplot as plt

def main():
	# create data
	a_length = 100
	b_length = 100
	a = np.random.normal([5,3],1,[a_length,2])
	b = np.random.normal([0,0],1,[b_length,2])

	plt.figure(0)
	plt.title("Ground truth")
	plt.scatter(a[:,0],a[:,1],color='r')
	plt.scatter(b[:,0],b[:,1],color='b')
	plt.legend(["negative","positive"],loc=2)

	X = np.concatenate([a,b])
	y = np.concatenate([-np.ones([a_length,1]),np.ones([b_length,1])])
	w = np.ravel(np.concatenate([1*np.ones([a_length,1]),np.ones([b_length,1])]))
	# train SVM
	mod = svm.SVR()

	mod.fit(X,y,w)

	# test
	y_hat = mod.predict(X)

	print(y_hat)
	# evaluate
	'''
	plt.figure(1)
	plt.title("Predicted")
	plt.scatter(X[y_hat==-1,0],X[y_hat==-1,1],color='r')
	plt.scatter(X[y_hat==1,0],X[y_hat==1,1],color='b')
	'''

	plt.show()

if __name__ == '__main__':
	main()
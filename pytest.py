# pytest.py

import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pp
import sim
import nasa

class Args:
	pass

def corr(a,b):
	return np.convolve(a,np.flip(b,axis=0))

def num_features(dat):
	return np.shape(dat)[1]

def linear_dependence(dat,lag):
	N = num_features(dat)
	dep = np.abs(np.dot(dat.T,dat))
	for i in range(1,lag):
		dep += np.abs(np.dot(dat[i:,:].T,dat[:-i,:]))

	return dep

def print_mat(mat):
	for row in mat:
		print("\t".join(["{0:.0f}".format(elem) for elem in row]))

def proportional_random_select(arr):
	c_arr = np.cumsum(arr)
	#print(c_arr)
	c_arr = c_arr/c_arr[-1]
	
	u = np.random.random()
	i = 0
	while u > c_arr[i]:
		i += 1

	return i

# num < 2 doesn't make sense!
def greedy_cand(dep,num):
	dep = np.abs(dep)
	for i in range(len(dep)):
		dep[i,i] = 0

	max_idx = np.unravel_index(dep.argmax(),dep.shape)
	idxs = []
	print_mat(dep)
	print()
	dep[max_idx[0],max_idx[1]] = 0
	dep[max_idx[1],max_idx[0]] = 0
	print_mat(dep)
	print()
	idxs.append(max_idx[0])
	idxs.append(max_idx[1])
	for i in range(2,num):
		dep_sum = np.sum(dep[idxs,:],axis=0)
		best = dep_sum.argmax()
		for idx in idxs:
			dep[best,idx] = 0
			dep[idx,best] = 0

		idxs.append(dep_sum.argmax())
		print_mat(dep)
		print()

	return idxs

def greedy_random_cand(dep,num):
	dep = np.abs(dep)
	for i in range(len(dep)):
		dep[i,i] = 0

	idx0 = proportional_random_select(np.sum(dep,axis=0))
	#print(idx0)
	idxs = [idx0]
	for i in range(1,num):
		#print_mat(dep)
		#print()
		dep_sum = np.sum(dep[idxs,:],axis=0)
		selected = proportional_random_select(dep_sum)
		#print(selected)
		for idx in idxs:
			dep[selected,idx] = 0
			dep[idx,selected] = 0

		idxs.append(selected)

	#print_mat(dep)
	#print()

	return idxs


args = Args()

args.filename = "../data/turbofan/train_FD001.txt"
args.linesep = "\n"
args.elemsep = " "
args.dataset = "TURBOFAN"
args.datatype = "ss"
data,explanations = nasa.main(args)

data,__ = pp.split(data,"TIMEWISE")
dat = data[0]
#print(dat)
dat,dat_mean,dat_std = pp.normalize(dat,return_mean_std=True)
#print(dat)
#print(dat_mean)
#print(dat_std)

'''
max_ord = 20

for i,feat_1 in enumerate(dat.T):
	for j,feat_2 in enumerate(dat.T[i:]):
		jx = j+i

		y = corr(feat_1,feat_2)
		mid = int(len(y)/2)
		y = y[mid:]

		plt.figure()
		plt.title("corr({0:d},{1:d})".format(i,jx))
		plt.plot(y[:max_ord])

plt.show()
'''


#res = np.dot(dat.T,dat)
#print_mat(res)
print(explanations)
dep = linear_dependence(dat,5)
print_mat(dep)
dep = (dep.T + dep)*0.5
#print_mat(dep)
for i in range(10):
	print(sorted(greedy_random_cand(dep,6)))

#for row in res:
#	print("\t".join(["{0:.0f}".format(elem) for elem in row]))


#from itertools import chain, combinations

'''
def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

def generate_subsets(N):
	a = list(map(list,powerset(range(N))))
	for elem in a:
		yield elem
'''
#g = generate_subsets(3)
#for i in range(5):
#	print(g.next())

'''
a = 5

def f():
	print("f: ")
	print(a)
	#a = 3
	#print(a)


def g():
	global a
	print("g: ")
	print(a)
	a = 3
	print(a)

def h():
	print("h:")
	a = 3
	print(a)

f()
h()
f()
g()
f()
'''
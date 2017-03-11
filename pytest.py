# pytest.py

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
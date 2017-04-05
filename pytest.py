# pytest.py

import numpy as np
import matplotlib.pyplot as plt

num_examples = 1000
num_dim = 50

#E = np.random.normal(0,1,[num_dim,num_dim])
#E = np.dot(E.T,E)
#print(np.linalg.det(E))
a = np.random.normal(0,1,[num_examples,num_dim])

U,S,V = np.linalg.svd(a)

S_energy = np.cumsum(S)
S_energy = S_energy/S_energy[-1]

plt.figure()
plt.plot(S_energy)
plt.show()
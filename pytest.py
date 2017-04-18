# pytest.py

import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(0,1,[100,2])
print(np.max(a,axis=0)*np.array([0.001,1]))
#plt.imshow(a,interpolation='none')
#plt.hist(a,100)
#plt.show()
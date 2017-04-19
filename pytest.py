# pytest.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

a = np.random.normal(0,1,[100,50])
#print(np.std(a,axis=1))
'''
f,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(a[:,0])
#axarr[1].plot(a[:,1])
axarr[1].imshow(a.T,interpolation='none')
axarr[1].axis([0,100,0,10])
'''
gs = gridspec.GridSpec(3,3)
ax = plt.subplot(gs[:2,:])
ax.imshow(a.T)
#plt.hist(a,100)
plt.show()
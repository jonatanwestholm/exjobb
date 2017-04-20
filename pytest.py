# pytest.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

a = np.random.normal(0,1,[100,50])
#print(np.std(a,axis=1))

f,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(a[:,0])
axarr[0].legend(["white noise"],loc="upper left",bbox_to_anchor=(1,1))
#axarr[1].plot(a[:,1])
axarr[1].imshow(a.T,interpolation='none')
axarr[1].axis([0,100,0,10])
#plt.hist(a,100)
plt.show()
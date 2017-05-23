# pytest.py

import numpy as np
import matplotlib.pyplot as plt

r = 0.9
f = lambda x: x
a = np.logspace(-3,3,100)

plt.plot(a)
plt.legend(["logspace"],loc=3)
plt.show()
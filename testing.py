# testing.py

import sim
import numpy as np

A = np.array([[0.5,-0.2,0,0],[0,0,0.5,-0.2]])
C = np.array([[1,0],[0,1]])

print(sim.varma_sim(C,A,100))
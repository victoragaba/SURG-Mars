import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

n = 1000
samples = []

for i in range(n):
    v = stats.norm.rvs(size=3)
    while np.linalg.norm(v) < 0.00001:
        v = stats.norm.rvs(size=3)
    v /= np.linalg.norm(v)
    samples.append(v)
    
fig = plt.figure()

import functions as fn
import numpy as np

start0 = np.zeros(3)
start1 = np.ones(3)
others = []
for _ in range(4):
    # create random array for t, normalised to 1
    t = fn.unit_vec(np.random.rand(3))

    # create random array for p, normalised to 1
    direc = fn.unit_vec(np.random.rand(3))
    p = fn.starting_direc(t, direc)

    # get true params for synthetic test
    add2start = fn.tp2sdr(t, p)[0]
    others.append(add2start)

starts = [start0, start1]
starts.extend(others)

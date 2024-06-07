import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from obspy.taup import TauPyModel
import function_repo as fr
from collections import defaultdict
import importlib
np.random.seed(1029)


# Implementing core algorithms faster
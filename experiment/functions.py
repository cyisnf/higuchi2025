import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import font_manager
from scipy.interpolate import griddata
import models
import warnings

warnings.resetwarnings()
warnings.simplefilter("ignore", RuntimeWarning)
row = 3
col = 2


def tab2data(arr_ct):
    row, col = arr_ct.shape
    x = []
    y = []
    for r in range(row):
        for c in range(col):
            val = int(arr_ct[r, c])
            x += [r] * val
            y += [c] * val
    return x, y


def data2tab(row, col, x, y):
    arr_ct = np.zeros([row, col])
    for i in range(len(x)):
        arr_ct[x[i], y[i]] += 1
    return arr_ct


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

#%%
#This loads the numerical array for transfer function from transfer_plot.py
ps = np.load("/Users/alisha/Documents/Vec_DM/PS.npy")

#Now define x_star

xsFIN = 500
xsIN = 0.001
xsnum = 500
xs = xsIN * (xsFIN/xsIN) ** ((np.arange(xsnum)) / xsnum)
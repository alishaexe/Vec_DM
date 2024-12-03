import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

#%%
#This loads the numerical array for transfer function from transfer_plot.py
size = 16
ps = np.load("/Users/alisha/Documents/Vec_DM/PS.npy")

#Now define x_star

xsFIN = 500
xsIN = 0.001
xsnum = 500
xs = xsIN * (xsFIN/xsIN) ** ((np.arange(xsnum)) / xsnum)

plt.loglog(xs, ps, label=r'$x^2_\star|T|^2$', color='indigo')
plt.xlabel(r'$x_\star$', fontsize=size)
plt.ylabel(r'$x^2_\star|T|^2$', fontsize=size)
plt.title('')
plt.grid(True) 
plt.legend(fontsize=13)
plt.show()

#%%
def sin(theta):
    return np.sin(theta)
def cos(theta):
    return np.cos(theta)

#These transfer functions come from the mathematica
#notebook integrals_07nov and are  from "marcos file"

def TE(y, xs):
    return sin(xs*y)/(xs*y)

#transfer function late A
def TLA(y, xs):
    # sh = sin(1/2)
    # ch = cos(1/2)
    
    # c1a = -sh*cos(xs)+(ch+0.5*sh)*(sin(xs)/xs)
    # c2a = ch*cos(xs)+(sh-0.5*ch)*(sin(xs)/xs)
    
    # res = 1/np.sqrt(y)*(c1a*cos((y**2)/2)+c2a*sin((y**2)/2))
    
    res = (2*sin(xs)*cos((1-y**2)/2)+(-2*xs*cos(xs)+sin(xs))*sin((1-y**2)/2))/(2*xs*np.sqrt(y))
    return res

#transfer function late B
def TLB(y, xs):    
    # c1b = sin((xs**2)/2)/(xs**1.5)*(1+sin(xs**2)/(2*xs**2))
    # c2b = cos((xs**2)/2)/(xs**1.5)*(1-sin(xs**2)/(2*xs**2))
    
    # res = 1/np.sqrt(y)*(c1b*cos(y**2/2)+c2b*sin(y**2/2))
    
    res = (-cos((3*xs**2-y**2)/2)+cos((xs**2+y**2)/2)+4*xs**2 * sin((xs**2+y**2)/2))/(4*xs**(7/2)*np.sqrt(y))
    return res

def T(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return np.where(
        condition_xs,
        np.where(condition_y1, TE(y, xs), TLA(y, xs)),  # xs < 1: check y < 1
        np.where(condition_yxs, TE(y, xs), TLB(y, xs))  # xs >= 1: check y < xs
    )


ymo = np.linspace(0.001,55,500)
def tcomp(y, xs):
    res =  xs**2*np.abs(T(y, xs))**2
    return res

test = tcomp(ymo, xs)
plt.loglog(xs, test)
plt.grid(True)
plt.show
#%%
plt.loglog(xs, test, label = "Marco")
plt.loglog(xs, ps, label="Numerical", color='indigo')
plt.xlabel(r'$x_\star$', fontsize=size)
plt.ylabel(r'$x^2_\star|T|^2$', fontsize=size)
plt.title('')
plt.grid(True) 
# plt.xlim(0.01, 100)
# plt.ylim(1e-6, 10)
plt.legend(fontsize=13)
plt.show()
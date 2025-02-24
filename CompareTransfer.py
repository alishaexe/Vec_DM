import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

from scipy.interpolate import interp1d

#%%
#This loads the numerical array for transfer function from transfer_plot.py
size = 16
ps = np.load("/Users/alisha/Documents/Vec_DM/datafiles/PS.npy")
ps2 = np.load("/Users/alisha/Documents/Vec_DM/datafiles/ps2.npy")
ps = ps2
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
def smooth(x):
    res = (0.0001*x**2)/(100 + x**2) + (0.00510204*x**2)/(1+0.364431*x**3)
    return res


xvals = np.logspace(np.log10(xsIN), np.log10(xsFIN), 1000)

pscurve = np.array(list(map(smooth, xvals)))

# plt.figure(figsize=(5, 3))
plt.loglog(xs, ps, label=r'Numerical', color='black')
plt.loglog(xvals, pscurve, '--',color = 'red', label = "Fitting function")
plt.xlabel(r'$x_\star$', fontsize=size)
plt.ylabel(r'$x^2_\star|T|^2$', fontsize=size)
plt.title('')
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.xlim(0.001, 100)
plt.ylim(1e-8, 1e-2)
plt.legend(fontsize = 11)
# plt.savefig('Plots/numeric_fitfunc.png', bbox_inches='tight')
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
    res = (2*sin(xs)*cos((1-y**2)/2)+(-2*xs*cos(xs)+sin(xs))*sin((1-y**2)/2))/(2*xs*np.sqrt(y))
    return res

#transfer function late B
def TLB(y, xs):    
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


def tcomp(y, xs):
    res =  xs**2*np.abs(T(y, xs))**2
    # res =  np.abs(T(y, xs))
    return res

#This is GT's approximation with the extra parts to make it one whole equation
def Ttot(y, xs):
    # xs = xsIN * (xsFIN/xsIN) ** ((i-1) / xsnum)
    ch=np.cos(1/2)
    sh= np.sin(1/2)
    
    c1a= -sh*np.cos(xs)+(ch+1/2*sh)*np.sin(xs)/xs
    c1b= np.sin(xs**2/2)/(xs**(3/2))*(1+np.sin(xs**2)/(2*xs**2))
    c2a= ch*np.cos(xs)+(sh-1/2*ch)*np.sin(xs)/xs
    c2b= np.cos(xs**2/2)/(xs**(3/2))*(1-np.sin(xs**2)/(2*xs**2))
    
    t1 = (1+xs)**4/((1+xs)**4+y**4)*np.sin(xs*y)/(xs*y)
    t2 = y**4/((1+xs)**4+y**4)*(1/np.sqrt(y)*(1/(1+xs**4)*c1a + xs**4/(1+xs**4)*c1b)*np.cos(y**2/2))
    t3 = y**4/((1+xs)**4+y**4)*(1/np.sqrt(y)*(1/(1+xs**4)*c2a + xs**4/(1+xs**4)*c2b)*np.sin(y**2/2))
    return xs**2*((t1+t2+t3))**2


# test1 = np.array(list(map(lambda args: tcomp(*args), )))
marco = tcomp(np.exp(4), xs)
GT = Ttot(np.exp(4), xs)
# plt.loglog(xs, marco)
# plt.grid(True)
# plt.show
#%%
# plt.loglog(xs, test, label = "Gianmassimo")
# plt.figure(figsize=(5, 3))
plt.loglog(xs, ps, label="Numerical", color='indigo')
plt.loglog(xs, marco, label="Analytical TF", color = "green")
plt.xlabel(r'$x_\star$', fontsize=size)
plt.ylabel(r'$x^2_\star|T|^2$', fontsize=size)
plt.title('')
plt.grid(True) 
# plt.xlim(0.001, 500)
plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.7) 
plt.xlim(0.001, 500)
plt.ylim(1e-8, 1e-2)
# plt.ylim(1e-6, 10)
plt.legend(fontsize = 11)
# plt.savefig('/Users/alisha/Documents/Vec_DM/Plots/numeric_analytic.png', bbox_inches='tight')
plt.show()
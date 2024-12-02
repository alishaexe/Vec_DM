import matplotlib.pyplot as plt
import time
import jax
import numpy as np
# import math
import jax.numpy as jnp
from jax import jit, grad, vmap, device_put, jacobian
from jax.scipy.integrate import trapezoid
#%%
pi = jnp.pi

@jit
def sin(theta):
    return jnp.sin(theta)

@jit
def cos(theta):
    return jnp.cos(theta)

#%%

#These transfer functions come from the mathematica
#notebook integrals_07nov and are  from "marcos file"

@jit
def TE(y, xs):
    return sin(xs*y)/(xs*y)

#transfer function late A
@jit
def TLA(y, xs):
    return 1/(2*xs*jnp.sqrt(y))*(2*sin(xs)*cos((1-y**2)/2)+(-2*xs*cos(xs)+sin(xs))*sin((1-y**2)/2))

#transfer function late B
@jit
def TLB(y, xs):
    return (-cos((3*xs**2-y**2)/2)+cos((xs**2+y**2)/2)+4*xs**2 * sin((xs**2+y**2)/2))/(4*xs**(7/2)*jnp.sqrt(y))

@jit
def T(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return jnp.where(
        condition_xs,
        jnp.where(condition_y1, TE(y, xs), TLA(y, xs)),  # xs < 1: check y < 1
        jnp.where(condition_yxs, TE(y, xs), TLB(y, xs))  # xs >= 1: check y < xs
    )  

## These correspond to the same derivative functions in the mathematica file
@jit
def TEdy(y, xs):
    return (xs*y * cos(xs*y)-sin(xs*y))/(xs*y**2)

@jit
def TLAdy(y, xs):
    t1 = 2* cos((1-y**2)/2)*(2*xs*y**2*cos(xs)-(1+y**2)*sin(xs))
    t2 = sin((1-y**2)/2)*(2*xs*cos(xs)+(-1+4*y**2)*sin(xs))
    return 1/(4*xs*y**(3/2))*(t1+t2)

@jit
def TLBdy(y, xs):
    t1 = cos((3*xs**2-y**2)/2)-2*y**2*sin((3*xs**2-y**2)/2)
    t2 = (1-8*xs**2*y**2)*cos((xs**2+y**2)/2)-2*(2*xs**2+y**2)*sin((xs**2+y**2)/2)
    return 1/(8*xs**(7/2)*y**(3/2))*(t1-t2)

@jit
def derivTdy(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return jnp.where(
        condition_xs,
        jnp.where(condition_y1, TEdy(y, xs), TLAdy(y, xs)),  # xs < 1: check y < 1
        jnp.where(condition_yxs, TEdy(y, xs), TLBdy(y, xs))  # xs >= 1: check y < xs
    )

@jit
def f(y, u, v, xs):
    res = T(y, u*xs)*T(y,v*xs)-((y**2*derivTdy(y, u*xs)*derivTdy(y,v*xs))/((u**2*xs**2+y**2)*(v**2*xs**2+y**2)))
    return res

# @jit
# def derivT(y, xs):
#     return jacobian(lambda y_: T(y_, xs))(y)

# @jit
# def f(y, u, v, xs):
#     res = T(y, u*xs)*T(y,v*xs)-((y**2*derivT(y, u*xs)*derivT(y,v*xs))/((u**2*xs**2+y**2)*(v**2*xs**2+y**2)))
#     return res

@jit
def IT_cos(y, u, v, xs):
    return y*cos(xs*y)*f(y, u, v, xs)

@jit
def IT_sin(y, u, v, xs):
    return y*sin(xs*y)*f(y, u, v, xs)

# @jit
# def IT(yend, u, v, xs):#should be 0.01 -> np.inf
#     yv = jnp.linspace(0.01, yend, 2000)
#     cos_integral = trapezoid(IT_cos(yv, u, v, xs), yv)
#     sin_integral = trapezoid(IT_sin(yv, u, v, xs), yv)
#     return (jnp.real(cos_integral))**2 + (jnp.real(sin_integral))**2

@jit
def IT(yend, u, v, xs):#should be 0.01 -> np.inf
    yi = jnp.linspace(0.01, yend/2, 7000)
    yf = jnp.linspace(yend/2, yend, 7000)
    cos_integral = trapezoid(IT_cos(yi, u, v, xs), yi)+trapezoid(IT_cos(yf, u, v, xs), yf)
    sin_integral = trapezoid(IT_sin(yi, u, v, xs), yi)+trapezoid(IT_sin(yf, u, v, xs), yf)
    return (jnp.real(cos_integral))**2 + (jnp.real(sin_integral))**2

#%%
yein = 20
yefin = 150
yenum = 500

yet = yein*(yefin/yein)**(jnp.arange(yenum+1)/yenum)

int1 = lambda g: IT(g, 10, 9, 0.6)
int2 = lambda g: IT(g, 10, 9, 0.1)
int3 = lambda g: IT(g, 10, 9, 1.3)

res1 = jnp.array(list(map(int1, yet)))
res2 = jnp.array(list(map(int2, yet)))
res3 = jnp.array(list(map(int3, yet)))
#%%
plt.loglog(yet, res1)
plt.title("scen 1")
# plt.xscale("log")
# plt.yscale("log")
plt.show()

plt.loglog(yet, res2)
plt.title("scen 2")
plt.show()

plt.loglog(yet, res3)
plt.title("scen 3")
# plt.xlim(20, 120)
# plt.ylim(1.3e-4, 3.5e-4)
# plt.xscale("log")
# plt.yscale("log")
plt.show()

#%%
start = time.time()

xsin = 0.2
xsfin = 1000
xsnum = 200

xt = xsin * (xsfin/xsin) ** ((jnp.arange(xsnum+1)) / xsnum)


tin = 0.01
tfin = 100
tnum=15

tt=tin*(tfin/tin)**((jnp.arange(tnum+1))/tnum)

tfactor = (tfin/tin)**(1/tnum)-1

sinn = -1
sfin = 1
snum=15

st = sinn+ (jnp.arange(snum+1))/snum*(sfin-sinn)

sfactor = 1/snum * (sfin-sinn)

@jit
def kk(i,j): #Tested with testkk and it prints 10201 array elements so works
    res = (tt[i]*(2+tt[i])*(st[j]**2-1))**2/((1-st[j]+tt[i])*(1+st[j]+tt[i]))**2
    return res

@jit
def OmegIntegrand(t, s, xs):
    u = (t+s+1)/2
    v = (t-s+1)/2
    res = IT(50, u, v, xs)
    # res2 = kk(i,j)*res*tt[i]
    return res

@jit
def OmegGW(m,i,j):
    res = (kk(i,j)*OmegIntegrand(tt[i],st[j],xt[m]))*tt[i]
    return res

m = jnp.arange(xsnum+1)


def compute_sum(m):
    iind = jnp.arange(tnum + 1)
    jind = jnp.arange(snum + 1)
    
    omegl = lambda i, j: OmegGW(m, i, j)
    # Broadcast computation across all i and j
    grid = jnp.array(jnp.meshgrid(iind, jind)).T.reshape(-1,2)
    igrid = grid[:,0]
    jgrid = grid[:,1]
    results = vmap(omegl)(igrid,jgrid)
    return jnp.sum(results)

tabres1 = vmap(compute_sum)(m)


@jit
def fintab(m):
    res = 0.5*1/96/pi**4*tfactor*sfactor*xt[m]**4*tabres1[m]
    return res

mt = jnp.arange(xsnum+1)
# Compute results
hope = vmap(fintab)(mt)
# hope = np.array(list(map(fintab, m)))
end = time.time()
mins = (end-start)/60
print(mins, "Minutes")

#Last time with 7000 in integrations and 200 x vals it took 8 mins

te = np.array(hope)
#%%

plt.loglog(xt, hope)
plt.loglog(x3, y3)
# plt.axline((x1,y1),(x2,y2), color = 'r')
plt.xlabel(r"$x_\star$", fontsize = 16)
# plt.ylabel(r"$\frac{M^4_{Pl}}{H^4_I}\Omega_{GW}$")
plt.ylabel(r"$\Omega_{GW}$")
plt.axhline(y=5e-6, color='r', linestyle='--', label="y = 0.5")
# plt.title("method 1")
# # plt.ylim(1e-5, 3e-4)
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/Vec_DM/JaxOmegGW.png', bbox_inches='tight')
plt.show()


#%%

def func(x):
    return 1e-4*(x)**2.5

x1, x2 = 0.2, 0.5
x3 = jnp.linspace(0.2, 0.8, 10)
y1, y2 = func(x1), func(x2)
y3 = func(x3)
# y1, y2 = 1e-7, 1.2e-4

plt.loglog(xt, te)
plt.loglog(x3, y3)
# plt.axline((x1,y1),(x2,y2), color = 'r')
plt.xlabel(r"$x_\star$", fontsize = 16)
# plt.ylabel(r"$\frac{M^4_{Pl}}{H^4_I}\Omega_{GW}$")
plt.ylabel(r"$\Omega_{GW}$")
plt.axhline(y=5e-6, color='r', linestyle='--', label="y = 0.5")
# plt.title("method 1")
# # plt.ylim(1e-5, 3e-4)
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/Vec_DM/JaxOmegGW.png', bbox_inches='tight')
plt.show()

#%%
from scipy.interpolate import UnivariateSpline
#%%
ym = np.log10(te)
xm = np.log10(xt)
spline = UnivariateSpline(xm, ym, s=0.5)

ysmoo = 10**spline(xm)

plt.figure(figsize=(8, 5))
# plt.loglog(xt, te, label='Original Data', alpha=0.7)
plt.loglog(xt, ysmoo, label='Smoothed Curve', linewidth=2)
plt.legend()
plt.show()
#%%
plt.loglog(xt, ysmoo, label = "Python")
plt.loglog(xdat, ydat, '--', markersize=5, label='Data')
plt.legend()
plt.grid(True)

#%%
data = [
    [0.2, 1.68575e-6], [0.218547, 2.09599e-6], [0.238813, 2.60489e-6], 
    [0.260959, 3.28666e-6], [0.285159, 4.17816e-6], [0.311603, 5.20304e-6],
    [0.340499, 6.28363e-6], [0.372075, 7.53625e-6], [0.406578, 9.18055e-6], 
    [0.444282, 1.12799e-5], [0.485482, 1.35966e-5], [0.530502, 1.5915e-5], 
    [0.579697, 1.89958e-5], [0.633455, 2.30602e-5], [0.692197, 2.5934e-5], 
    [0.756387, 2.94267e-5], [0.82653, 3.50427e-5], [0.903177, 3.80892e-5], 
    [0.986931, 4.40926e-5], [1.07845, 4.77267e-5], [1.17846, 5.48365e-5],
    [1.28774, 5.84677e-5], [1.40716, 6.52861e-5], [1.53765, 7.09824e-5], 
    [1.68025, 7.60901e-5], [1.83606, 8.20633e-5], [2.00632, 8.83102e-5], 
    [2.19238, 9.38795e-5], [2.39569, 9.80632e-5], [2.61785, 1.01594e-4], 
    [2.86061, 1.04401e-4], [3.12588, 1.06852e-4], [3.41576, 1.08298e-4], 
    [3.73251, 1.09993e-4], [4.07864, 1.10491e-4], [4.45687, 1.11707e-4], 
    [4.87017, 1.12119e-4], [5.3218, 1.12159e-4], [5.8153, 1.12411e-4], 
    [6.35458, 1.12717e-4], [6.94386, 1.12775e-4], [7.58779, 1.12734e-4], 
    [8.29143, 1.12672e-4], [9.06033, 1.12485e-4], [9.90052, 1.12151e-4], 
    [10.8186, 1.12031e-4], [11.8219, 1.11643e-4], [12.9182, 1.11308e-4], 
    [14.1161, 1.10919e-4], [15.4252, 1.10489e-4], [16.8556, 1.09984e-4], 
    [18.4187, 1.09314e-4], [20.1267, 1.0859e-4], [21.9931, 1.07804e-4], 
    [24.0326, 1.06799e-4], [26.2612, 1.05714e-4], [28.6965, 1.044e-4], 
    [31.3577, 1.02936e-4], [34.2656, 1.0123e-4], [37.4431, 9.93014e-5], 
    [40.9154, 9.70536e-5], [44.7096, 9.45595e-5], [48.8557, 9.17463e-5], 
    [53.3862, 8.86179e-5], [58.3369, 8.50337e-5], [63.7467, 8.11187e-5], 
    [69.6582, 7.67686e-5], [76.1178, 7.21193e-5], [83.1765, 6.70595e-5], 
    [90.8898, 6.16338e-5], [99.3183, 5.59066e-5], [108.528, 4.99202e-5], 
    [118.593, 4.38206e-5], [129.59, 3.77656e-5], [141.608, 3.1894e-5], 
    [154.739, 2.63081e-5], [169.089, 2.11025e-5], [184.769, 1.64908e-5], 
    [201.903, 1.26811e-5], [220.627, 9.85718e-6], [241.086, 7.9531e-6], 
    [263.443, 6.73748e-6], [287.873, 6.01661e-6], [314.568, 5.66114e-6], 
    [343.739, 5.57368e-6], [375.615, 5.44249e-6], [410.448, 4.87802e-6], 
    [448.51, 4.03016e-6], [490.102, 3.27005e-6], [535.551, 2.64619e-6], 
    [585.214, 2.11117e-6], [639.483, 1.8003e-6], [698.785, 1.83914e-6], 
    [763.586, 1.59903e-6], [834.396, 1.36485e-6], [911.772, 1.24355e-6], 
    [996.324, 9.94898e-7], [1088.72, 8.61792e-7], [1189.68, 8.04384e-7], 
    [1300.0, 7.45641e-7]
]

# Split data into x and y vectors
xdat, ydat = np.array(data).T
#%%

fgw = 7.5e-6
def fOmeg(y):
    res = y*1e-6
    return res

def freq(f):
    res = f*fgw
    return res

omegdat = np.array(list(map(fOmeg, ydat)))
fdat = np.array(list(map(freq, xdat)))

plt.loglog(fdat, omegdat)
plt.show()

omegme = np.array(list(map(fOmeg, ysmoo)))
fme = np.array(list(map(freq, xt)))

plt.loglog(fme, omegme)
plt.loglog(fdat, omegdat)
plt.show

#%%
yr = 365*24*60*60 #in seconds
H0 = 100*0.67*10**(3)/(3.086*10**(22)) #1/seconds
# H0 = 3.24e-18 #Debikas value
#setting h = 0.67
pi = np.pi
c = 3e8
fetstar = 10**(-2)
fi = 0.4*10**(-3)

#For the LISA mission they have designed the
#arms to be length L = 2.5*10^(6)
T = 3*yr
snr5 = 5
#L = 2.5e9
L = 25/3

c = 3e8
fetstar = 10**(-2)
fi = 0.4*10**(-3)

fLisa = 1/(2*pi*L)
ffmin = 10**(-5)
ffmax = 445
elmin = np.log10(ffmin)
elmax = np.log10(ffmax)
###############
#Change this value for how many 'steps' you want in the range of values

itera = 100
nitera = 10
##########

elminL = (np.log10(ffmin))
elmaxL = (np.log10(10**(-0.95)))
elminet = np.log10(1.6)
elmaxet = (np.log10(ffmax))
ntmin = -9/2
ntmax = 9/2
step = (ntmax-ntmin)/itera


P = 12
A = 3


def P_acc(f):
    res = A**2 *(1e-15)**2 * (1+(0.4e-3 / f)**2)*(1+(f/8e-3)**4)*(2*pi*f)**(-4)*(2*pi*f/c)**2
    return res

def P_ims(f):#* (1e-12)**2 after P
    res = P**2 * (1e-12)**2 *(1+(2e-3/f)**4)*(2*pi*f/c)**2
    return res

def N_aa(f):
    con = 2*pi*f*L
    res = 8 * (np.sin(con))**2 * (4*(1+np.cos(con)+(np.cos(con))**2)*P_acc(f)+(2+np.cos(con))*P_ims(f))
    return res

def R(f):
    res = 16*(np.sin(2*pi*f*L))**2  * (2*pi*f*L)**2 * 9/20 * 1/(1+0.7*(2*pi*f*L)**2)
    return res

def S_n(f):
    res = N_aa(f)/R(f)
    return res

def Ohms(f):
    const = 4*pi**2/(3*H0**2)
    res = const *f**3*S_n(f)
    return res

freqvals = np.logspace(elminL, elmaxL, 200)   
sigvals = np.array(list(map(Ohms, freqvals)))


#%%
fbplo = np.load("ftablisa.npy")

plt.figure(figsize=(6, 7))
plt.loglog(freqvals, sigvals, label = "Nominal Curve", color = "indigo", linewidth=1.5)
plt.loglog(np.exp(fbplo[:,0]), np.exp(fbplo[:,1]), label = "BPLS curve", color = "lime", linewidth=1.5)
plt.loglog(fme, omegme, label="PythonCurve")
plt.loglog(fdat, omegdat, label = "DataCurve")
plt.ylabel(r"$\Omega_{GW}$")
plt.xlabel(r"$f (Hz)$")
plt.legend()
plt.grid(True)
plt.savefig('/Users/alisha/Documents/Vec_DM/OverlayOmegGW.png', bbox_inches='tight')
plt.show()

#%%




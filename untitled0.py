

import matplotlib.pyplot as plt
import time
import jax
import math
from joblib import Parallel, delayed
import jax.numpy as jnp
from jax import jit, grad, vmap
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

@jit
def IT_cos(y, u, v, xs):
    return y*cos(xs*y)*f(y, u, v, xs)

@jit
def IT_sin(y, u, v, xs):
    return y*sin(xs*y)*f(y, u, v, xs)

@jit
def IT(yend, u, v, xs):#should be 0.01 -> np.inf
    y = jnp.linspace(0.01, yend, 1000)
    cos_integral = trapezoid(IT_cos(y, u, v, xs), y)
    sin_integral = trapezoid(IT_sin(y, u, v, xs), y)
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
xsfin = 300
xsnum = 250

xt = xsin * (xsfin/xsin) ** ((jnp.arange(xsnum+1)) / xsnum)

tin = 0.01
tfin = 300
tnum=100

tt=tin*(tfin/tin)**((jnp.arange(tnum+1))/tnum)

tfactor = (tfin/tin)**(1/tnum)-1

sinn = -1
sfin = 1
snum=100

st = sinn+ (jnp.arange(snum+1))/snum*(sfin-sinn)

sfactor = 1/snum * (sfin-sinn)

@jit
def kk(i,j):
    res = (tt[i]*(2+tt[i])*(st[j]**2-1))**2/((1-st[j]+tt[i])*(1+st[j]+tt[i]))**2
    return res

@jit
def OmegIntegrand(t, s, xs):
    u = (t+s+1)/2
    v = (t-s+1)/2
    res = IT(50, u, v, xs)
    return res

@jit
def OmegGW(m,i,j):
    res = (kk(i,j)*OmegIntegrand(tt[i],st[j],xt[m]))*tt[i]
    return res

m = jnp.arange(xsnum+1)

@jit
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

@jit
def fintab(m):
    test = vmap(compute_sum)(m)
    res = 0.5*1/96/pi**4*tfactor*sfactor*xt[m]**4*test[m]
    return res

# Compute results
tabres1 = fintab(m)

end = time.time()
mins = (end-start)/60
print(mins, "Minutes")

#%%
plt.loglog(xt, tabres1)
plt.xlabel(r"$x_\star$", fontsize = 16)
plt.ylabel(r"$\frac{M^4_{Pl}}{H^4_I}\Omega_{GW}$")
plt.title("method 1")
# plt.ylim(1e-5, 3e-4)
plt.grid(True)
plt.savefig('/Users/alisha/Documents/Vec_DM/JaxOmegGW.png', bbox_inches='tight')
plt.show()
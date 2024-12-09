import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, device_put, jacobian
from jax.scipy.integrate import trapezoid

#%%
PTA = np.loadtxt("/Users/alisha/Documents/Vec_DM/sensitivity_curves_NG15yr_fullPTA.txt",delimiter=',', skiprows=1)

pi = jnp.pi
H0 = 100*0.67*10**(3)/(3.086*10**(22))
fpta = 1e-8

snr = 5
year = 365*24*60*60
T = 5* year

freqs = PTA[:, 0]
h_c = PTA[:, 1] # characteristic strain
S_eff = PTA[:, 2] #strain power spectral density
Omega_GW = PTA[:, 3] #calculated using H0=67.4 km/s/Mpc = 67.4*1e3/1/3e22 s^-1 to match the paper 
#%%

@jit
def pls(a):
    res = ((freqs/fpta)**a/(Omega_GW))**2
    return res



@jit
def integrate(a):
    I1 = jnp.trapezoid(pls(a), freqs)
    res = snr/(jnp.sqrt(2*T*I1))
    return res

alpha = jnp.arange(-8, 9, 1) #this list from -8 to 8

Atab = vmap(integrate)(alpha)


fmin = 2e-10
fmax = 2e-7
bins = 5000
fstep = (fmax-fmin)/bins

frequency = jnp.arange(fmin, fmax,fstep)


@jit
def ftest(f):
    return Atab*(f/fpta)**alpha


fvals = vmap(ftest)(frequency)


@jit
def maxtab():
    maxed = (jnp.max(fvals, axis = 1))
    return maxed

iit = jnp.arange(len(frequency))

PLS = maxtab()

plt.loglog(frequency, PLS)
plt.loglog(freqs, Omega_GW)
plt.grid(True)
# plt.ylim(1e-10,1e-1)
plt.show()
#%%

otab = np.load("/Users/alisha/Documents/Vec_DM/jaxOmeg.npy")
xt, te = otab[:,0], otab[:,1]
fgw1 = 1e-11

def case1(y):
    res = y*1e-6
    return res

def freq1(f):
    res = f*fgw1
    return res

def OmegAnalytical(f):
    a1 = 2
    a2 = 3
    
    n1 = 2.6
    n2 = -0.05
    n3 = -2
    
    f1 = 1
    f2 = 90
    
    t1 = (f/f1)**n1
    t2 = (1+(f/f1)**a1)**((-n1+n2)/a1)
    t3 = (1+(f/f2)**a2)**((-n2+n3)/a2)
    return 1.25e-4*t1*t2*t3

analytical = np.array(list(map(OmegAnalytical, xt)))

plt.loglog(xt, analytical)
plt.show()

#For case 1 using the analytical fit we get
anaomeg1 = np.array(list(map(case1, analytical)))
anafreq1 = np.array(list(map(freq1, xt)))

#%%
plt.figure(figsize=(6, 7))
# plt.loglog(frequency, PLS, label = "PLS", color = "orangered")
plt.loglog(freqs, Omega_GW, label = r"Nominal Nanograv", color = "indigo")
# plt.loglog(anafreq1, case1(te), label ="numerical")
plt.loglog(freqs, Omega_GW*1e-2, label = r"Nominal SKA", color = "blue")
plt.loglog(anafreq1, anaomeg1, label = "Case 1 (analytical)", color = "black")
plt.ylabel(r"$\Omega_{GW}$")
plt.xlabel(r"$f (Hz)$")
plt.legend(loc = 9, fontsize = 12)
plt.grid(True)
plt.savefig('/Users/alisha/Documents/Vec_DM/Plots/OverlayOmegGW_skanano.png', bbox_inches='tight')
plt.show()
#%%

#Changing this to strain to compare
#S_h = 3H0**2/2pi**2*Omega_gw / f**3
#h_c = sqrt(f*s_h)=sqrt(3H0**2/2pi**2*Omega/f**2)

# def strain(f):
#     res = np.sqrt(3*H0**2/(2*pi**2)*Omega_GW/f**2)
#     return res

# hc = vmap(strain)(xt)
# plt.loglog(frequency, hc)

nomhc = np.sqrt(3*H0**2/2*pi**2*Omega_GW / freqs**2)
plshc = np.sqrt(3*H0**2/2*pi**2*PLS / frequency**2)
anahc = np.sqrt(3*H0**2/2*pi**2*anaomeg1/ anafreq1**2)


plt.figure(figsize=(6, 7))
# plt.loglog(frequency, plshc, label = "PLS")
plt.loglog(freqs, nomhc, label = r"$nom$")
# plt.loglog(freqs, h_c, label ="data")
plt.loglog(anafreq1, anahc, label = "Case 1 (analytical)", color = "black")
plt.ylabel(r"$h_{c}$")
plt.xlabel(r"$f (Hz)$")
plt.legend(loc = 9, fontsize = 12)
plt.grid(True)
# plt.savefig('/Users/alisha/Documents/Vec_DM/Plots/OverlayOmegGW_analytical1.png', bbox_inches='tight')
plt.show()

#%%

@jit
def bpls(n1, n2):
    s = 10
    res = ((freqs/fpta)**n1 * (1/2+1/2*(freqs/fpta)**s)**(-(n1-n2)/s))
    return res

@jit
def integrateb(a, b):
    I1 = jnp.trapezoid((bpls(a, b)/Omega_GW)**2, freqs)
    res = snr/(jnp.sqrt(2*T*I1))
    return res

alpha = jnp.arange(-8, 9, 1) #this list from -8 to 8
beta = jnp.arange(-8, 9, 1)

powers = jnp.array(jnp.meshgrid(alpha, beta)).T.reshape(-1,2)
n1 = powers[:,0]
n2 = powers[:,1]

Btab = vmap(integrateb)(n1, n2)
# testtab = integrateb(n1, n2)

fmin = 2e-10
fmax = 2e-7
bins = 5000
fstep = (fmax-fmin)/bins

frequency = jnp.arange(fmin, fmax,fstep)


@jit
def ftest(f):
    s=10
    # return Btab*bpls(n1,n2)
    return Btab*((f/fpta)**n1 * (1/2+1/2*(f/fpta)**s)**(-(n1-n2)/s))


fvals = vmap(ftest)(frequency)


@jit
def maxtab():
    maxed = (jnp.max(fvals, axis = 1))
    return maxed

# iit = jnp.arange(len(frequency))

BPLS = maxtab()

plt.loglog(frequency, BPLS, label = "BPLS", color = "lime")
plt.loglog(freqs, Omega_GW, color = "indigo", label = "Nominal")
plt.loglog(frequency, PLS, label = "PLS", color = "orangered")
plt.legend()
plt.grid(True)
# plt.ylim(1e-10,1e-1)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad
import math
from scipy.misc import derivative
from scipy.integrate import simpson, quadrature, romberg, fixed_quad
from scipy import stats
from scipy.integrate import qmc_quad
#%%
pi = np.pi


#%%

## This function for T uses the approximations rather than marcos so left out
# def T(y, xs):
#     if xs <= 1:
#         if y <= 1:
#             return 1+y**2/4-y**4/4
#         if y > 1:
#             return 1/np.sqrt(y)
#     if xs >1:
#         if y <= xs:
#             t1 = (1+(1-xs**2)*y**2)/(xs**4+8*y**3)
#             t2 = (1+8*xs*y**4)
#             return t1/t2
#         if y > xs:
#             return 1/(xs**(3/2)*np.sqrt(y))

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
    return 1/(2*xs*np.sqrt(y))*(2*sin(xs)*cos((1-y**2)/2)+(-2*xs*cos(xs)+sin(xs))*sin((1-y**2)/2))

#transfer function late B
def TLB(y, xs):
    return (-cos((3*xs**2-y**2)/2)+cos((xs**2+y**2)/2)+4*xs**2 * sin((xs**2+y**2)/2))/(4*xs**(7/2)*np.sqrt(y))

def T(y, xs):
    if xs < 1:
        if y < 1:
            return TE(y, xs)
        else:
            return TLA(y, xs)
    else:
        if y < xs:
            return TE(y, xs)
        else:
            return TLB(y, xs)

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

## Here originally I had python doing the derivative for each term
## Have since switched to using the derivatives given from the mathematica file

# def derivT(y, xs):
#     return derivative(lambda y: T(y, xs), y, dx=1e-6)

# def f(y, u, v, xs):
#     res = T(y, u*xs)*T(y,v*xs)-((y**2*derivT(y, u*xs)*derivT(y,v*xs))/((u**2*xs**2+y**2)*(v**2*xs**2+y**2)))
#     return res


## These correspond to the same derivative functions in the mathematica file
def TEdy(y, xs):
    return (xs*y * cos(xs*y)-sin(xs*y))/(xs*y**2)

def TLAdy(y, xs):
    t1 = 2* cos((1-y**2)/2)*(2*xs*y**2*cos(xs)-(1+y**2)*sin(xs))
    t2 = sin((1-y**2)/2)*(2*xs*cos(xs)+(-1+4*y**2)*sin(xs))
    return 1/(4*xs*y**(3/2))*(t1+t2)

def TLBdy(y, xs):
    t1 = cos((3*xs**2-y**2)/2)-2*y**2*sin((3*xs**2-y**2)/2)
    t2 = (1-8*xs**2*y**2)*cos((xs**2+y**2)/2)-2*(2*xs**2+y**2)*sin((xs**2+y**2)/2)
    return 1/(8*xs**(7/2)*y**(3/2))*(t1-t2)

# def derivTdy(y, xs):
#     if xs<1:
#         if y < 1:
#             return TEdy(y, xs)
#         else:
#             return TLAdy(y, xs)
#     else:
#         if y < xs:
#             return TEdy(y, xs)
#         else:
#             return TLBdy(y, xs)
            
#         # if y >= xs:
#         #     return TLBdy(y, xs)

def derivTdy(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return np.where(
        condition_xs,
        np.where(condition_y1, TEdy(y, xs), TLAdy(y, xs)),  # xs < 1: check y < 1
        np.where(condition_yxs, TEdy(y, xs), TLBdy(y, xs))  # xs >= 1: check y < xs
    )

def f(y, u, v, xs):
    res = T(y, u*xs)*T(y,v*xs)-((y**2*derivTdy(y, u*xs)*derivTdy(y,v*xs))/((u**2*xs**2+y**2)*(v**2*xs**2+y**2)))
    return res


def IT_cos(y, u, v, xs):
    return y*np.cos(xs*y)*f(y, u, v, xs)

def IT_sin(y, u, v, xs):
    return y*np.sin(xs*y)*f(y, u, v, xs)

def IT(u, v, xs):#should be 0.01 -> np.inf
    cres, cerr = qmc_quad(IT_cos, 0.01, np.inf, args=(xs, u, v))
    sres, serr = qmc_quad(IT_sin, 0.01, np.inf, args=(xs, u, v))
    return (np.real(cres))**2+(np.real(sres))**2

def OmegIntegrand(t, s, xs):
    u = (t+s+1)/2
    v = (t-s+1)/2
    t1 = ((t*(2+t)*(s**2-1))/((1-s+t)*(1+s+t)))**2
    t2 = IT(u, v, xs)
    return t1*t2

def OmegGW(xs):
    factor = xs**4/(96*pi**4) 
    res, err = dblquad(OmegIntegrand, -1, 1, lambda t: 0, lambda t: np.inf, args=(xs,))
    return factor*res


## As of yet have not ran OmegGW() since it takes so long to integrate, might make the
## integrals smaller to check it does work correctly
# final = OmegGW(0.7)

#%%

xsin = 0.2
xsfin = 300
xsnum = 25

xt = xsin * (xsfin/xsin) ** ((np.arange(xsnum+1)) / xsnum)

tin = 0.01
tfin = 300
tnum=10

tt=tin*(tfin/tin)**((np.arange(tnum+1))/tnum)

tfactor = (tfin/tin)**(1/tnum)-1

sinn = -1
sfin = 1
snum=10

sfactor = 1/snum * (sfin-sinn)

## Redefining the function so that we can do the tests of the y_end value

def ITy(yend, u, v, xs):
    ## qmc_quad doesn't take arguments like the normal quad so to contain
    ## the other parameters and only integrate over y we use the lambda funcitons
    cosf = lambda y: IT_cos(y, u, v, xs)
    sinf = lambda y: IT_sin(y, u, v, xs)
    ## QMC_quad uses Quasi-Monte Carlo quadrature to compute the integrals
    cres, cerr = qmc_quad(cosf, 0.01, yend)
    sres, serr = qmc_quad(sinf, 0.01, yend)
    return (np.real(cres)**2+np.real(sres)**2)

yein = 20
yefin = 150
yenum = 100

yet = yein*(yefin/yein)**(np.arange(yenum+1)/yenum)

int1 = lambda g: ITy(g, 0.6, 10, 9)
int2 = lambda g: ITy(g, 0.1, 10, 9)
int3 = lambda g: ITy(g, 1.3, 10, 9)

res1 = np.array(list(map(int1, yet)))
res2 = np.array(list(map(int2, yet)))
res3 = np.array(list(map(int3, yet)))

#%%
plt.loglog(yet, res1)
plt.title("res1")
# plt.xlim(0,200)
# plt.ylim(1e-4,0.01)
plt.show()

plt.loglog(yet, res2)
plt.title("res2")
# plt.xlim(0,200)
# plt.ylim(1e-4,0.01)
plt.show()

plt.loglog(yet, res3)
plt.title("res3")
# plt.xlim(0,200)
# plt.ylim(1e-4,0.01)
plt.show()

#%%

## The plot from the previous bit did not give a stabilised beginning like mathematica
## So I copied the format of the mathematica file to check whether this had an effect
## It does not




########################################
## Mainly using the code from previous section, had used this to check
## pretty similar so going with stuff above
########################################
def Tp(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return np.where(
        condition_xs,
        np.where(condition_y1, TE(y, xs), TLA(y, xs)),  # xs < 1: check y < 1
        np.where(condition_yxs, TE(y, xs), TLB(y, xs))  # xs >= 1: check y < xs
    )

def Tpdy(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return np.where(
        condition_xs,
        np.where(condition_y1, TEdy(y, xs), TLAdy(y, xs)),  # xs < 1: check y < 1
        np.where(condition_yxs, TEdy(y, xs), TLBdy(y, xs))  # xs >= 1: check y < xs
    )   
        
def Tq(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return np.where(
        condition_xs,
        np.where(condition_y1, TE(y, xs), TLA(y, xs)),  # xs < 1: check y < 1
        np.where(condition_yxs, TE(y, xs), TLB(y, xs))  # xs >= 1: check y < xs
    )
        
def Tqdy(y, xs):
    condition_xs = xs < 1
    condition_y1 = y < 1
    condition_yxs = y < xs

    # Use np.where for vectorized branching
    return np.where(
        condition_xs,
        np.where(condition_y1, TEdy(y, xs), TLAdy(y, xs)),  # xs < 1: check y < 1
        np.where(condition_yxs, TEdy(y, xs), TLBdy(y, xs))  # xs >= 1: check y < xs
    )  
        
def Tc(y, xs, u, v):
    return y*cos(y*xs)*(Tp(y, u*xs)*Tq(y, v*xs)-(y**2*Tpdy(y, u*xs)*Tqdy(y, v*xs))/((u**2*xs**2+y**2)*(v**2*xs**2+y**2)))

def Ts(y, xs, u, v):
    return y*sin(y*xs)*(Tp(y, u*xs)*Tq(y, v*xs)-(y**2*Tpdy(y, u*xs)*Tqdy(y, v*xs))/((u**2*xs**2+y**2)*(v**2*xs**2+y**2)))

def intc(yend, xs, u, v):
    cres = romberg(IT_cos, 0.01, yend, args=(xs, u, v))
    sres = romberg(IT_sin, 0.01, yend, args=(xs, u, v))
    return (np.real(cres))**2+(np.real(sres))**2


int1 = lambda g: intc(g, 0.6, 10, 9)
int2 = lambda g: intc(g, 0.1, 10, 9)
int3 = lambda g: intc(g, 1.3, 10, 9)

res1 = np.array(list(map(int1, yet)))
res2 = np.array(list(map(int2, yet)))
res3 = np.array(list(map(int3, yet)))

#%%

## Change res3 to res2 or res1 etc to get the plot you want 
plt.loglog(yet, res2)
# plt.xlim(0,50)
plt.show()

#%%

def intf(yend, xs, u, v):
    f = lambda x: IT_cos(x, u, v, xs)
    g = lambda x: IT_sin(x, u, v, xs)
    a = 0.01
    b = yend
    N = 1000
    n = 1000 #use n*N+1 points to plot smoothly?
    
    x = np.linspace(a, b, N+1)
    cosp = f(x)
    sinp = g(x)
    
    X = np.linspace(a, b, n*N+1)
    cosP = f(X)
    sinP = g(X)
    
    dx = (b-a)/N
    x_left = np.linspace(a, b-dx, N)
    # x_mid = np.linspace(dx/2, b-dx/2, N)
    # x_right = np.linspace(dx, b, N)
    
    mid_rem_fp = np.sum(f(x_left)*dx)
    mid_rem_gp = np.sum(g(x_left)*dx)
    return mid_rem_fp**2 + mid_rem_gp**2

int1 = lambda g: intf(g, 0.6, 10, 9)
int2 = lambda g: intf(g, 0.1, 10, 9)
int3 = lambda g: intf(g, 1.3, 10, 9)

res1 = np.array(list(map(int1, yet)))
res2 = np.array(list(map(int2, yet)))
res3 = np.array(list(map(int3, yet)))

#%%
plt.loglog(yet, res1)
plt.title("res1 left")
plt.show()


plt.loglog(yet, res2)
plt.title("res2 left")
plt.show()


plt.loglog(yet, res3)
plt.title("res3 left")
plt.show()

#%%

def intf(yend, xs, u, v, point):
    f = lambda x: IT_cos(x, u, v, xs)
    g = lambda x: IT_sin(x, u, v, xs)
    a = 0.01
    b = yend
    N = 1000
    n = 1000 #use n*N+1 points to plot smoothly?
    
    x = np.linspace(a, b, N+1)
    cosp = f(x)
    sinp = g(x)
    
    X = np.linspace(a, b, n*N+1)
    cosP = f(X)
    sinP = g(X)
    
    dx = (b-a)/N
    if point == 1: #this gives the left points
        x_point = np.linspace(a, b-dx, N)
    if point == 2: #this gives the midpoints
        x_point = np.linspace(dx/2, b-dx/2, N)
    if point == 3: #this gives the right midpoints
        x_point = np.linspace(dx, b, N)
    
    rem_fp = np.sum(f(x_point)*dx)
    rem_gp = np.sum(g(x_point)*dx)
    return rem_fp**2 + rem_gp**2

int1l = lambda g: intf(g, 0.6, 10, 9, 1)
int2l = lambda g: intf(g, 0.1, 10, 9, 1)
int3l = lambda g: intf(g, 1.3, 10, 9, 1)

res1l = np.array(list(map(int1l, yet)))
res2l = np.array(list(map(int2l, yet)))
res3l = np.array(list(map(int3l, yet)))

int1m = lambda g: intf(g, 0.6, 10, 9, 2)
int2m = lambda g: intf(g, 0.1, 10, 9, 2)
int3m = lambda g: intf(g, 1.3, 10, 9, 2)

res1m = np.array(list(map(int1m, yet)))
res2m = np.array(list(map(int2m, yet)))
res3m = np.array(list(map(int3m, yet)))

int1r = lambda g: intf(g, 0.6, 10, 9, 3)
int2r = lambda g: intf(g, 0.1, 10, 9, 3)
int3r = lambda g: intf(g, 1.3, 10, 9, 3)

res1r = np.array(list(map(int1r, yet)))
res2r = np.array(list(map(int2r, yet)))
res3r = np.array(list(map(int3r, yet)))

#%%

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(yet,res1l,'b')
plt.title('Left Riemann sum of scen1')

plt.subplot(1,3,2)
plt.plot(yet,res1m,'b')
plt.title('mid Riemann sum of scen1')

plt.subplot(1,3,3)
plt.plot(yet,res1r,'b')
plt.title('right Riemann sum of scen1')

plt.show()

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(yet,res2l,'b')
plt.title('Left Riemann sum of scen2')

plt.subplot(1,3,2)
plt.plot(yet,res2m,'b')
plt.title('mid Riemann sum of scen2')

plt.subplot(1,3,3)
plt.plot(yet,res2r,'b')
plt.title('right Riemann sum of scen2')

plt.show()

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(yet,res3l,'b')
plt.title('Left Riemann sum of scen3')

plt.subplot(1,3,2)
plt.plot(yet,res3m,'b')
plt.title('mid Riemann sum of scen3')

plt.subplot(1,3,3)
plt.plot(yet,res3r,'b')
plt.title('right Riemann sum of scen3')

plt.show()
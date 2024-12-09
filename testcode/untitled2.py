
xs1=0.01
plt.plot(y, sol(xs1), label = "og sol")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('')
plt.grid(True) 
# plt.legend(fontsize=13)
plt.show()


plt.plot(y, sol(xs1), label = "og sol")
plt.plot(y, Ttot(y,xs1), label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.xlim(0,8)
plt.title('xs = {xs}'.format(xs = xs1))
plt.grid(True) 
plt.legend(fontsize=13)
plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import math
ana = hilbert(sol(xs1))
en = np.abs(ana)


plt.figure(figsize=(12, 6))
plt.plot(y, ana, label='Oscillating Function', color='blue')
plt.plot(y, en, label='Envelope', color='red', linestyle='--')
plt.title('Oscillating Function and Its Envelope')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

#%%
peaks, blank = find_peaks(sol(xs1))

xpeak = y[peaks]
ypeak = sol(xs1)[peaks]

func = interp1d(xpeak, ypeak, kind = "cubic", fill_value = "extrapolate")
sfunc = func(y)

plt.plot(y, sol(xs1),':', label=r'$numerical$', color='indigo')
plt.plot(y, sfunc, 'r--', label = "smoothed curve over the peaks")
plt.scatter(xpeak, ypeak)
plt.xlim(0, 7)

#%%
# from scipy.optimize import curve_fit
def poly(x, xs, a,b):
    return (1)/((x)**(0.5))+xs
    

# test, _ = curve_fit(poly, y, sol(xs1))

# curve = poly(y, *test)

# plt.plot(y,curve)
# plt.plot(y, sol(xs1),':', label=r'$numerical$', color='indigo')
# plt.plot(y, sfunc, 'r--', label = "smoothed curve over the peaks")
# plt.scatter(xpeak, ypeak)
# plt.xlim(0, 7)
# plt.ylim(-1,2)

#%%
def Tfin(x, xs):
    s1 = 1/(1+np.exp(4*(x-(1+xs))))
    t2 = 1-(xs**1.5*x**2)/2
    t4 = 1/(x**0.5 + 1)
    res = s1*t2+(1-s1)*t4
    # res = 1/(1+(x/xs)**0.5)
    return res
y = np.linspace(0,10,1000)
def poly(x, xs, a,b):
    return (1)/((x)**(0.5)+1)+xs
def fact(m):
    return math.factorial(m)
def cosx(x):
    return 1-(xs1*x)**2/fact(2)
def test(x):
    res = (1+xs1)**3/((1+xs1)**3+x**3)*cosx(x)+x**4/((1+xs1)**4+x**4)*poly(x,xs1,1,1)
    return res
boop = sol(xs1)
meep = poly(y,xs1, 1, -0.03)
keep = cosx(y)
koop = test(y)
moop = Tfin(y,xs1)
plt.plot(y, meep, label = "meep")
plt.plot(y, moop, label = "moop")
# plt.plot(y, koop, label = "koop")
plt.plot(y, sol(xs1),':', label=r'$numerical$', color='indigo')
# plt.plot(y, sfunc, 'r--', label = "smoothed ")
# plt.scatter(xpeak, ypeak)
plt.legend()
plt.xlim(0, 7)
plt.ylim(-1,1.5)
#%%
def Tfin(x, xs):
    s1 = 1/(1+np.exp(2*(x-(1+xs))))
    t2 = 1-(xs*x)**2/2+(xs*x)**4/24
    t4 = 1/(x**0.5 + 1)
    res = s1*t2+(1-s1)*t4
    # res = 1/(1+(x/xs)**0.5)
    return res
#%%
gradm = np.gradient(meep)
grads = np.gradient(sfunc)
test2 = y[gradm!=grads]

test3 = np.vstack((y.T,meep.T, sfunc.T)).T

#%%
eq = sol(xs2)
peaks, blank = find_peaks(sol(xs2))

xpeak = y[peaks]
ypeak = eq[peaks]

func = interp1d(xpeak, ypeak, kind = "cubic", fill_value = "extrapolate")
sfunc = func(y)
xs=0.02
# beep = poly(y,1,-0.03)
boop = Tfin(y, xs)
plt.plot(y,boop, label="approx")
# plt.plot(y, boop)
# plt.plot(y, sfunc, label = "app")
plt.plot(y, sol(xs2), label = "original")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xs))
plt.grid(True) 
plt.legend(fontsize=13)
# plt.ylim(-1,2)
plt.savefig('/Users/alisha/Documents/T{xs}.png'.format(xs=xs), bbox_inches='tight')
plt.show()

#%%

# def Tfin(x, xs):
#     t1 = (xs-1)**3/((xs-1)**3+x**3)
#     t2 = 1-((xs-1)*x)**2/2+((xs-1)*x)**4/24
#     t3 = x**3/((xs-1)**3+x**3)
#     t4 = 0.5/((x)**0.5+1)
#     res = t1*t2+t3*t4
#     return res

def Tfin(x, xs):
    s1 = 1/(1+np.exp(6*(x-(1/xs))))
    t2 = 1-(xs**1.5*x**2)/2
    
    t4 = 1/(x*xs + 1)
    res = s1*t2+(1-s1)*t4
    # res = 1/(1+(x*xs**0.75))
    return res

### doesnt work
# def Tmac(x, xs):
#     a = 10
#     xbar = x-1/xs
#     # s1 = 2/(a**2*xbar**2+2*a*xbar**2+1)
#     s1 = 1/(a**2*(x-1/xs)**2+2)
#     t2 = 1-(xs**1.5*x**2)/2
    
#     t4 = 1/(x*xs + 1)
#     res = s1*t2+(1-s1)*t4
#     return res
    

def suffer(x, xs):
    t1 = (1+xs)/((1+xs)+x)
    t2 = x**2/(1+xs+x**10)
    
    return t1-t2
    
xs3 = 1.5
# moop = poly(y,0.8, -0.1)
moop = Tfin(y,1.5)
maap = suffer(y,1.5)
plt.plot(y,moop, label = "moop")
plt.plot(y,maap, label = "maap")

plt.plot(y, sol(xs3),'--', label = "og sol")
plt.plot(y, 0.5/y**0.5-0.05,'-.', label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xs3))
plt.grid(True) 
plt.legend(fontsize=13)
plt.ylim(-0.5,1.5)
plt.xlim(0,7)
plt.show()

#%%
# def Tfin(x, xs):
#     t1 = (xs-1)**3/((xs-1)**3+x**3)
#     t2 = 1-(x)**2/(2*xs**4)
#     t3 = x**4/((1+xs)**4+x**4)
#     t4 = 0.5/(xs**1.5*(x)**0.5+1)*xs**4/(1+xs)**4
#     res = t1*t2+t3*t4
#     # res = t2
#     return res

def Tfin(x, xs):
    s1 = 1/(1+np.exp(6*(x-(1/xs))))
    t2 = 1-(xs**1.5*x**2)/2
    
    t4 = 1/(x*xs + 1)
    res = s1*t2+(1-s1)*t4
    # res = 1/(1+(x*xs**0.75))
    return res

# def Tfin(x, xs):
#     s1 = 1/(1+np.exp(4*(x-(1+xs))))
#     t2 = 1-(xs**1.5*x**2)/2
#     t4 = 1/(x**0.5 + 1)
#     res = s1*t2+(1-s1)*t4
#     # res = 1/(1+(x/xs)**0.5)
#     return res
def suffer(x, xs):
    t1 = (1+xs)/((1+xs)+x)
    t2 = x**2/(1+xs**0.5+x**10)
    
    return t1-t2
xsv=0.5
moop = suffer(y,xsv)
plt.plot(y,moop, label="approx")
plt.plot(y, sol(xsv),'--', label = "original")
# plt.plot(y, 0.6/y**0.5-0.3,'-.', label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xsv))
plt.grid(True) 
plt.legend(fontsize=13)
# plt.ylim(-0.5,1.25)
plt.xlim(0,7)
xs_int = int(xsv * 100)
# plt.savefig('/Users/alisha/Documents/T{xs:03d}.png'.format(xs=xs_int), bbox_inches='tight')
plt.show()

#%%
import numpy as np
from scipy.integrate import quad
import numdifftools as nd
from findiff import FinDiff

def TfinA(x, xs):
    s1 = 1/(1+np.exp(4*(x-(1+xs))))
    t2 = 1-(xs**1.5*x**2)/2
    t4 = 1/(x**0.5 + 1)
    res = s1*t2+(1-s1)*t4
    return res


def TfinB(x, xs):
    s1 = 1/(1+np.exp(6*(x-(1/xs))))
    t2 = 1-(xs**1.5*x**2)/2
    
    t4 = 1/(x*xs + 1)
    res = s1*t2+(1-s1)*t4
    return res

# def integrateA(xs, ye):
#     dT_dy = nd.Derivative(lambda y: TfinA(y, xs))(ye)
    
#     term1 = (xs**2 / (xs**2 + ye**2)) * (np.absolute(dT_dy))**2
#     term2 = xs**2 * (np.absolute(Tfin(ye, xs)))**2
#     return (term1 + term2) / xs

# def integrateB(xs, ye):
#     dT_dy = nd.Derivative(lambda y: TfinB(y, xs))(ye)
    
#     term1 = (xs**2 / (xs**2 + ye**2)) * (dT_dy)**2
#     term2 = xs**2 * (Tfin(ye, xs))**2
#     return  (term1 + term2) / xs

# def integral(ye):
#     I1, _ = quad(integrateA, 0, 1, args=(ye,))
#     I2, _ = quad(integrateB, 1, ye, args=(ye,))
#     result = ye*(I1 + I2)
#     return result

# energy = integral(10)
# print(energy)

#%%

#Now going to try using the bernstein equations
#Helpful to first make a nice binomial function

def bino(n,r):
    return math.comb(n, r)


def bern(n,r,x):
    r = 0
    res0 = bino(n,r)*x**r*(1-x)**(n-r)
    r = 1
    res1 = bino(n,r)*x**r*(1-x)**(n-r)
    r = 2
    res2 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 3
    # res3 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 4
    # res4 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 5
    # res5 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 6
    # res6 = bino(n,r)*x**r*(1-x)**(n-r)
    res = res1+res2#+res3+res4+res5+res6
    # res = bino(n,r)*x**r*(1-x)**(n-r)
    return res

def berntest(n,x):
    A=np.zeros(len(x))
    for i in range(len(x)):
        a=0
        for r in range(n+1):
            a += bino(n,r)*x**r*(1-x)**(n-r)
            
            
        A[i]= sum(a)
    return A

def testcurve(n,x,xs):
    a = berntest(n,x)
    res = a*1+(1-a)*(1/(xs*x+1))
    return res

def newcurve(n,r,x,xs):
    a=bern(n,r,x)
    # A=0
    # a=0
    # for r in range(n+1):
    #     A=bern(n,r,x)
    #     a+=A
        
    #res = alpha * k1 +[1-alpha]*k2
    #For use we use k1 = 1, and k2 = 1/sqrt(y)? maybe try 1/(xs*y+1)
    res = a*1+(1-a)*(1/(xs*x+1))
    return res
xsv=0.5
y = np.linspace(0.001,10,10)
ythou = np.linspace(0.001,10,1000)


test = newcurve(4,1,y/max(y),xsv).T
# t2 = testcurve(3,y/max(y),xsv).T
moop = suffer(y,xsv)
plt.plot(y,test, label="test")
# plt.plot(y,t2, label="test2")

# plt.plot(ythou, sol(xsv),'--', label = "original")
# plt.plot(y, 1/(xsv*y+1),'-.', label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xsv))
plt.grid(True) 
plt.legend(fontsize=13)
# plt.ylim(-0.5,1.25)
# plt.xlim(0,7)
xs_int = int(xsv * 100)
# plt.savefig('/Users/alisha/Documents/T{xs:03d}.png'.format(xs=xs_int), bbox_inches='tight')
plt.show()

#%%

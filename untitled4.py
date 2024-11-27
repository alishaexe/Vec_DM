import numpy as np
import matplotlib.pyplot as plt
import math

#Now going to try using the bernstein equations
#Helpful to first make a nice binomial function

def bino(n,r):
    # print(r)
    return math.comb(n, r)


def bern(n,r,x):
    # res = 0
    # for r in range(n+1):
    #     res += bino(n,r)*x**r*(1-x)**(n-r)
    #     print(res)
    # r = 0
    # res0 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 1
    # res1 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 2
    # res2 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 3
    # res3 = bino(n,r)*x**r*(1-x)**(n-r)
    # r = 4
    # res4 = bino(n,r)*x**r*(1-x)**(n-r)
    # res = res1+res2+res3+res4
    res = bino(n,r)*x**r*(1-x)**(n-r)
    # print(res)
    return res#res1+res2+res3+res4

def alpha(n,x):
    r = np.arange(0,n+1)
    res = np.array(list(map(lambda r_i, y_i: bern(n, r_i, y_i), r, y))) 
    return (res)


def newcurve(n,x,xs):
    
    a=sum(np.array(list(map(lambda x_i: alpha(n,x_i),x))))
    print(a)
    #res = alpha * k1 +[1-alpha]*k2
    #For use we use k1 = 1, and k2 = 1/sqrt(y)? maybe try 1/(xs*y+1)
    res = a*1+(1-a)*(1/(xs*x+1))
    return res

xsv=0.5
y = np.linspace(0.001,10,10)

test = newcurve(2,y/max(y),xsv).T
# moop = suffer(y,xsv)
plt.plot(y,test, label="test")
# plt.plot(y, sol(xsv),'--', label = "original")
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
import math
import numpy as np

# Binomial function
def bino(n, r):
    return math.comb(n, r)

# Bernstein basis polynomial
def bern(n, r, x):
    return bino(n, r) * (x ** r) * ((1 - x) ** (n - r))

# Function to compute new curve using a sum over Bernstein polynomials with different weights
def newcurve(n, x, xs, A):
    total = 0  # Initialize sum
    for r in range(n + 1):
        a = bern(n, r, x)
        # Use a weight A[r] for each Bernstein term
        total += (a * 1 + (1 - a) * (1 / (xs * x + 1)))
    a = bern(n, r, x)
    total = (a * 1 + (1 - a) * (1 / (xs * x + 1)))
    return total

# Example usage
xsv = 0.5
x = np.linspace(0, 1, 100)  # Replace with your actual x values
y = np.sin(2 * np.pi * x)  # Replace with your actual y values (normalized)
r = 6
# Coefficients A to weight the Bernstein polynomials (adjust these as needed)
A = np.random.rand(r+1)  # Example: random weights for the Bernstein polynomials

test = newcurve(r, x / max(x), xsv, A).T

# Optionally, you can plot or further analyze `test`
print(test)

plt.plot(x,y)
plt.plot(x, test)

#%%
#Now going to try using the bernstein equations
#Helpful to first make a nice binomial function

def bino(n,r):
    return math.comb(n, r)


def bern(n,r,x):
    res = bino(n,r)*x**r*(1-x)**(n-r)
    return res

def berntest(n,x):
    A=np.zeros(len(x))
    for i in range(len(x)):
        a=0
        for r in range(n+1):
            a += bino(n,r)*x**r*(1-x)**(n-r)
            
            
        A[i]= sum(a)
    return A

def alpha(l,m,n,x):
    #sum from i=0 to l of B_i^l+m+1
    a=0
    for i in range(l+1):
        a += bern(n,i,x)
        print(a)
        
    return
    



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
y = np.linspace(0,1,10)

test = newcurve(3,1,y/max(y),xsv).T
t2 = testcurve(3,y,xsv)
# moop = suffer(y,xsv)
plt.plot(y,test, label="test")
# plt.plot(y,t2, label="test2")

# plt.plot(y, sol(xsv),'--', label = "original")
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
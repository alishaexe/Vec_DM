import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
#%%
size = 16
#%%
# xs range
xsFIN = 500
xsIN = 0.001
xsnum = 500
xs = xsIN * (xsFIN/xsIN) ** ((np.arange(xsnum)) / xsnum)
ivals = np.arange(1, xsnum+1)
xsval = xsIN * (xsFIN/xsIN) ** ((ivals-1) / xsnum)
# diff equation
def diffeqs(y, z, xs_i):
    z1, z2 = z #letting z = vp(y)
    dz1_dy = z2  # So then z2 = vp'(y). And z1' = z2
    dz2_dy = - (2/y) * (1/(1 + (y**2 / xs_i**2))) * z2 - xs_i**2 * (1 + (y**2 / xs_i**2)) * z1
    return [dz1_dy, dz2_dy] #this is vp' and vp''

# the array for Ps
ps = np.zeros(xsnum)
#%%
# start = time.time()
# # this solves xs individually and computes ps
# for i, xsi in enumerate(xs):
#     # init cons vp(0.001) = 1, vp'(0.001) = 0
#     v0 = [1, 0]  # conditions on vp and vp'
#     yspan = (0.001, 55)  # range of y
#     yeval = np.linspace(0.001, 55, 1000)  # points  for evaluating
#     # solving
#     xsi = xsval[i]
#     sol = solve_ivp(diffeqs, yspan, v0, t_eval=yeval, args=(xsi,), method="Radau")
    
#     # Extract the solution for vp(y) at y = 55
#     vp_at_55 = sol.y[0, -1]  # Last point corresponds to y=55
    
#     #y axis 
#     ps[i] = xsval[i]**2 * vp_at_55**2

# end = time.time()
# forloop = end-start
# print("the for loop method took", forloop)
#%%
#This method takes about 0.02 seconds longer than the forloop. 
#So if end up needing more iterations will use this instead

start = time.time()
def solve_xs(xs_i):
    # Initial conditions: vp(0.001) = 1, vp'(0.001) = 0
    z0 = [1.0, 0.0]
    y_span = (0.001, 55)
    y_eval = np.linspace(0.001, 55, 1000)

    # Solve the ODE for this particular xs_i
    solution = solve_ivp(diffeqs, y_span, z0, t_eval=y_eval, args=(xs_i,), method="Radau")
    
    # Extract the solution for vp(y) at y = 55
    vp_at_55 = solution.y[0, -1]  # Last point corresponds to y=55

    # Return Ps[i] = xs[i]^2 * vp(55)^2
    return xs_i**2 * vp_at_55**2

ps = np.array(list(map(solve_xs, xs)))
end = time.time()
arrayt= end - start
print("the array method took", arrayt)
#%%
# plt.figure(figsize=(8, 8))
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
#%%
plt.figure(figsize=(5, 3))
plt.loglog(xs, ps,':', label=r'$x^2_\star|T|^2$', color='indigo')
# plt.loglog(xs, nonit)

plt.loglog(xvals, pscurve, color = 'black')
plt.xlabel(r'$x_\star$', fontsize=size)
plt.ylabel(r'$x^2_\star|T|^2$', fontsize=size)
plt.title('')
plt.grid(True) 
plt.xlim(0.001, 100)
# plt.legend(fontsize=13)
plt.show()


#%%
# peaks, blank = find_peaks(ps)

# xpeak = xs[peaks]
# ypeak = ps[peaks]

# func = interp1d(xpeak, ypeak, kind = "cubic", fill_value = "extrapolate")
# sfunc = func(xs)

# plt.loglog(xs, ps,':', label=r'$x^2_\star|T|^2$', color='indigo')
# plt.loglog(xs, sfunc, 'r--', label = "smoothed curve over the peaks")
# plt.scatter(xpeak, ypeak)
#%%

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
    return(t1+t2+t3)

y = np.linspace(0.001,10,500)

Tfin = Ttot(y, 0.01)
plt.plot(y, Tfin, label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('')
plt.grid(True) 
# plt.legend(fontsize=13)
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
    return 1/(2*xs*np.sqrt(y))*(2*sin(xs)*cos((1-y**2)/2)+(-2*xs*cos(xs)+sin(xs))*sin((1-y**2)/2))

#transfer function late B
def TLB(y, xs):
    return (-cos((3*xs**2-y**2)/2)+cos((xs**2+y**2)/2)+4*xs**2 * sin((xs**2+y**2)/2))/(4*xs**(7/2)*np.sqrt(y))

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


#%%
#Comparing the 2 transfer functions
def xstar(s, e, num):
    xs = s * (e/s) ** ((np.arange(num)) / num)
    return xs

xmode = xstar(0.001, 100, 1000)

ymo = np.linspace(0.001,100,1000)
def tcomp(y, xs):
    res =  y**2*np.abs(T(y, xs))**2
    return res

test = tcomp(xmode, ymo)
plt.loglog(xmode, test)
plt.grid(True)
plt.show
#%%
plt.loglog(xmode, test, label = "Marco")
plt.loglog(xs, ps, label="Numerical", color='indigo')
plt.xlabel(r'$x_\star$', fontsize=size)
plt.ylabel(r'$x^2_\star|T|^2$', fontsize=size)
plt.title('')
plt.grid(True) 
# plt.xlim(0.01, 100)
# plt.ylim(1e-6, 10)
plt.legend(fontsize=13)
plt.show()

#%%

y = np.linspace(0.01, 20, 1000)

def sol(xs):
    z0 = [1.0, 0.0]
    y_span = (0.001, 55)
    y_eval = np.linspace(0.01, 20, 1000)

    # Solve the ODE for this particular xs_i
    solution = solve_ivp(diffeqs, y_span, z0, t_eval=y_eval, args=(xs,), method="Radau")
    return solution.y[0]


#%%
xs1=0.01
numfunc=sol(xs1)

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
plt.title('xs = {xs}'.format(xs = xs1))
plt.grid(True) 
plt.legend(fontsize=13)
plt.show()

#%%
xs2 = 0.3
# plt.plot(y, sol(xs2), label = "og sol")
# plt.xlabel(r'$y$', fontsize=size)
# plt.ylabel(r'$T$', fontsize=size)
# plt.title('')
# plt.grid(True) 
# # plt.legend(fontsize=13)
# plt.show()


plt.plot(y, sol(xs2), label = "og sol")
plt.plot(y, Ttot(y,xs2), label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xs2))
plt.grid(True) 
plt.legend(fontsize=13)
plt.show()

#%%
xs3 = 1.5
plt.plot(y, sol(xs3),'--', label = "og sol")
plt.plot(y, Ttot(y,xs3),'-.', label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xs3))
plt.grid(True) 
plt.legend(fontsize=13)
plt.show()

#%%
xs4 = 20
plt.plot(y, sol(xs4),'-', label = "og sol")
plt.plot(y, Ttot(y,xs4),'--', label = "approximation")
plt.xlabel(r'$y$', fontsize=size)
plt.ylabel(r'$T$', fontsize=size)
plt.title('xs = {xs}'.format(xs = xs4))
plt.xlim(-0.1,6)
plt.grid(True) 
plt.legend(fontsize=13)
plt.show()
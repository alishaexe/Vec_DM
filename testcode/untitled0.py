import sympy as sp
from sympy import *
from sympy import Array, collect, simplify, trigsimp, Derivative, Add
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor
from einsteinpy.symbolic import RicciTensor, RicciScalar, EinsteinTensor, StressEnergyMomentumTensor
#%%
A, eta, phi = sp.symbols('A eta phi')
x = sp.symbols('x1 x2 x3')
a = sp.Function('a')(eta)
B = sp.Function('B')(*x)


# Define the metric components
g_00 = -a**2 * (1 + 2 * A)
g_0i = [a**2 * 2 * sp.diff(B, xi) for xi in x]  # Assuming B is a function of x_i


g = sp.zeros(4,4)
g[0,0]=g_00
for i in range(1,4):
    g[0,i]=g_0i[i-1]
    g[i,0]=g_0i[i-1]
    g[i,i]=a**2

syms = (eta, *x)
m_obj = MetricTensor(Array(g),syms)
#%%
phi = sp.Function('phi')(eta, *x)

T_obj = StressEnergyMomentumTensor(phi, m_obj)
#%%
delta = sp.Function('delta')(eta)
phi = sp.Function('phi')(eta)

g00 = -1/a**2 * (1-2*A)

inside = 1/2 * g00 * (sp.diff(phi, eta)+sp.diff(delta, eta))*(sp.diff(phi, eta)+sp.diff(delta, eta))

outside = -g_00 * inside
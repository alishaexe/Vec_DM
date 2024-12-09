import sympy as sp
from sympy.diffgeom import Manifold, Patch, CoordSystem
from einsteinpy.symbolic import MetricTensor, RicciTensor, ChristoffelSymbols
# Define the symbols and functions
t, x1, x2, x3, epsilon = sp.symbols('t x_1 x_2 x_3 epsilon')
a = sp.Function('a')(t)
B = sp.Function('B')(t, x1, x2, x3)
NN = sp.Function('NN')(t, x1, x2, x3)
zeta = sp.Function('zeta')(t, x1, x2, x3)

coords = t, x1, x2, x3
#%%

# Define the matrix g
g = sp.Matrix([
    [-1 - 2 * epsilon * NN, 
     a * epsilon * sp.diff(B, x1), 
     a * epsilon * sp.diff(B, x2), 
     a * epsilon * sp.diff(B, x3)],
    
    [a * epsilon * sp.diff(B, x1), 
     a**2 * (1 + 2 * epsilon * zeta), 
     a**2 * epsilon * sp.diff(0, x1, x2), 
     a**2 * epsilon * sp.diff(0, x1, x3)],
    
    [a * epsilon * sp.diff(B, x2), 
     a**2 * epsilon * sp.diff(0, x1, x2), 
     a**2 * (1 + 2 * epsilon * zeta), 
     a**2 * epsilon * sp.diff(0, x2, x3)],
    
    [a * epsilon * sp.diff(B, x3), 
     a**2 * epsilon * sp.diff(0, x1, x3), 
     a**2 * epsilon * sp.diff(0, x2, x3), 
     a**2 * (1 + 2 * epsilon * zeta)]
])

# Simplify each element of the matrix
g_simplified = g.applyfunc(sp.simplify)

# Display the simplified matrix
# sp.pprint(g_simplified, use_unicode=True)

g_series = g.applyfunc(lambda expr: expr.series(epsilon, 0, 3).removeO())

# Simplify the expanded series
g_simpser = sp.simplify(g_series)

# Pretty print the simplified matrix
sp.pprint(g_simpser, use_unicode=True)

#%%

# Define the metric tensor
metric = MetricTensor(g_simpser, coords)
metric.tensor()
# Compute the Christoffel symbols
ch = ChristoffelSymbols.from_metric(metric)
ch.tensor()
# Compute the Ricci tensor
ricci_tensor = RicciTensor.from_christoffels(ch)

# Display the Ricci tensor
ricci_tensor.tensor()
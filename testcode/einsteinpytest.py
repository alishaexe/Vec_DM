import sympy as sp
from sympy import Array, collect, simplify, trigsimp, Derivative, Add
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor
from einsteinpy.symbolic import RicciTensor, RicciScalar, EinsteinTensor
#%%
A, eta = sp.symbols('A eta')
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
m_obj.tensor()

ch = ChristoffelSymbols.from_metric(m_obj)

rm1 = RiemannCurvatureTensor.from_christoffels(ch)
rm1.tensor()
#%%
ric = RicciTensor.from_metric(m_obj)
ric.tensor()
#%%
R_0i = [ric.tensor()[0, i] for i in range(1, 4)]  # Accessing R_01, R_02, R_03

# If you want to simplify the results
R_0i_simplified = [simplify(comp) for comp in R_0i]

# Printing results
for i, value in enumerate(R_0i_simplified):
    print(f"R_0{i + 1} = {value}")
#%%
R = RicciScalar.from_riccitensor(ric)
R.simplify()
R.expr


#%%
einst = EinsteinTensor.from_metric(m_obj)
einst.tensor()
#%%
G_0i = [einst[0,i] for i in range(1,4)]
G_0i_simplified = [simplify(comp) for comp in G_0i]



#%%
for i in range(4):
    for j in range(4):
        if g[i, j] != 0:
            simplified_expr = simplify(g[i, j])
            collected_expr = collect(simplified_expr, [a, B, A])
            sp.pprint(collected_expr, use_unicode=True)
            
#%%
def remove_higher_order_terms(expr):
    # Collect all terms that are not of the form A * Derivative(B) or higher
    filtered_terms = []
    for term in expr.as_ordered_terms():
        # Check if the term contains A and a derivative of B
        if not (A in term.free_symbols and isinstance(term, Mul) and any(isinstance(arg, Derivative) and arg.has(B) for arg in term.args)):
            filtered_terms.append(term)

    # Return the simplified expression without the unwanted terms
    return simplify(Add(*filtered_terms))

fil = remove_higher_order_terms(g_0i)
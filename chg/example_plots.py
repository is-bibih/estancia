import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre as L, jv, yv, factorial, poch, gamma
#from scipy.special import hyp1f1 as M
#from mpmath import hyp1f1 as M
from ..functions.special import M, U

which_function = 'bessel'

num = 200
x = np.linspace(0, 10, num=num)

# -------------------
# function parameters
# -------------------

y = a = b = coef = comp_fun = legends = None
fmt_string = '-'

if which_function == 'laguerre':
    legends = ['HGC', 'Laguerre']
    # laguerre
    n = 3 # integer
    alpha = 2
    # hgc
    a = -n
    b = alpha+1
    coef = factorial(n) / poch(alpha + 1, n)
    comp_fun = laguerreL(n, alpha, x)

elif which_function == 'bessel':
    legends = ['aproximación con HGC', 'Bessel']
    a = -13
    b = 2.01
    coef = gamma(b) * np.exp(0.5*x) * ((0.5*b-a)*x)**(0.5-0.5*b)
    comp_fun = jv(b-1, np.sqrt(2*x*(b - 2*a)))

elif which_function == 'neumann':
    legends = ['aproximación con HGC', 'Neumann']
    M = U
    a = -13
    #x = x/np.abs(a) * 5
    b = 1
    coef = gamma(0.5*b - a + 0.5) * np.exp(0.5*x) * x**(0.5-0.5*b)
    comp_fun = np.cos(a*np.pi) * jv(b-1, np.sqrt(2*x*(b-2*a))) \
        - np.sin(a*np.pi) * yv(b-1, np.sqrt(2*x*(b-2*a)))

elif which_function == 'chg':
    x = np.linspace(0, 1, num=num)
    fmt_string = '-'

    num_a = 3
    a = np.linspace(-5, 5, num=num_a)
    b = np.linspace(1, 5, num=num_a)

    y = np.zeros([num, num_a**2])
    legends = [None] * num_a**2

    k = 0
    for i in range(a.size):
        for j in range(b.size):
            y[:, k] = hyperM(a[i], b[j], x)
            legends[k] = f'$a = {a[i]}, b = {b[j]}$'
            k += 1

# --------
# evaluate
# --------

if which_function != 'chg':
    y = hyperM(a, b, x) / coef

# ----
# plot
# ----

fig, ax = plt.subplots(1, 1)
ax.plot(x, y, fmt_string)
if which_function != 'chg':
    ax.plot(x, comp_fun)
ax.legend(legends)
plt.show()


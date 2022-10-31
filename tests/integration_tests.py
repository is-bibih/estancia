import numpy as np
from scipy.special import erf

from ..functions.integration import mc_integrate

x0 = -5
xf = 5
n = 1e6
func = lambda x: np.exp(-x**2)

mc_result = mc_integrate(x0, xf, func, n)
an_result = np.sqrt(np.pi) * erf(5)

print(mc_result - an_result)


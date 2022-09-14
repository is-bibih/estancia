import numpy as np
import matplotlib.pyplot as plt
from hypgeo_functions import X, L
from scipy.special import factorial
import time

num = 300
xf = 50

n = 8
m = 2

x = np.linspace(1, xf, num=num).reshape([1, -1])
yl = L(n, m, x)

# truncated sum
start_badsums = time.time()
yx, Cmn = X(m, n, x, method='bad sums')
end_badsums = time.time()

# converging sum
start_sums = time.time()
yx, Cmn = X(m, n, x, method='sums')
end_sums = time.time()

N = n + (m+1)/2
amn = factorial(n) * N**(m/2) / factorial(n+m)
bmn = 1/(factorial(n) * factorial(m) * N**(m/2) * Cmn)

coef_l = amn * np.exp(-x/2) * x**(m/2)
coef_x = bmn * np.exp(-x/2) * x**(m/2)

y_both = np.exp(-x/2) * x**(m/2) \
            * np.sqrt(amn**2 * yl**2 + bmn**2 * yx**2)

yl = yl * coef_l
yx = yx * coef_x

x, yl, yx, y_both = [xi.flatten() for xi in (x, yl, yx, y_both)]

fig, ax = plt.subplots(1,1)
ax.plot(x, yl, label='laguerre 1')
ax.plot(x, yx, label='laguerre 2')
ax.plot(x, y_both, label='envelope')
ax.legend()
#plt.show()

print(end_sums - start_sums)
print(end_badsums - start_badsums)


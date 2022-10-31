import numpy as np
import matplotlib.pyplot as plt

from ..functions.hypergeometric import X

x0 = 1
xf = 10
num = 300

x = np.linspace(x0, xf, num=num)
y_3_4 = X(3, 4, x).flatten()
y_30_4 = X(30, 4, x).flatten()
y_3_40 = X(3, 50, x).flatten()

fig, ax = plt.subplots(1, 1)
ax.plot(x, y_3_4, label=r'$m=3, m=4$')
#ax.plot(x, y_30_4, label=r'$m=30, m=4$')
ax.plot(x, y_3_40, label=r'$m=3, m=40$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$X^m_n$')
fig.legend()
plt.show()


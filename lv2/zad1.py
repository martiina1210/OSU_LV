import numpy as np
import matplotlib.pyplot as plt


x = [1.0, 3.0, 3.0, 2.0, 1.0]
y = [1.0, 1.0, 2.0, 2.0, 1.0]


plt.plot(x, y, color='green', linewidth=2)
plt.fill(x, y, color='lightgreen', alpha=0.5) 
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.title('Trapez s četiri točke')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

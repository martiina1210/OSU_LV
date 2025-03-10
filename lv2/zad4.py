import numpy as np
import matplotlib.pyplot as plt


crni = np.zeros((50, 50), dtype=int)  
bijeli = np.ones((50, 50), dtype=int)   


gore = np.hstack((crni, bijeli))  
dolje = np.hstack((bijeli, crni))  


slika = np.vstack((gore, dolje))


plt.imshow(slika, cmap='gray')
plt.axis([0,100,100,0])  
plt.show()

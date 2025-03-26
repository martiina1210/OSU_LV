import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("lv2/road.jpg")
img = img[:,:,0].copy()

plt.imshow(img,cmap="gray")
plt.show()




plt.imshow(img,cmap="gray",alpha=0.75)
plt.show()

x = img[:,160:320]
plt.imshow(x,cmap="gray")
plt.show()

rot = np.rot90(m=img,k=-1)
plt.imshow(rot,cmap="gray")
plt.show()

mirror = np.fliplr(img)
plt.imshow(mirror,cmap="gray")
plt.show()

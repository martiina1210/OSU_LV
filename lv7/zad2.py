import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("lv7/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

unique_colors = np.unique(img_array, axis=0)
print(f"Broj razlicitih boja u originalnoj slici: {len(unique_colors)}")

K = int(input("Unesite K: "))

# Primijeni K-means
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(img_array)

# Zamijeni RGB vrijednosti svakog piksela s njegovim najbližim centrom
centres = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[centres]

# Vrati natrag u oblik originalne slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title(f"Kvantizirana slika s K={K} boja")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

inertias = []
K_values = range(1, 9)

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_array)
    inertias.append(kmeans.inertia_)

# Prikaz grafa
plt.figure()
plt.plot(K_values, inertias, marker='o')
plt.xlabel("Broj grupa K")
plt.ylabel("Inertia (J)")
plt.title("Elbow metoda za određivanje optimalnog K")
plt.grid(True)
plt.show()

for k in range(K):
    mask = (centres == k).astype(np.uint8)
    mask_img = np.reshape(mask, (w, h))

    plt.figure()
    plt.title(f"Grupa {k+1}")
    plt.imshow(mask_img, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

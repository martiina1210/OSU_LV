import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('lv2/data.csv', delimiter=',',  skip_header=1)

broj_osoba = data.shape[0]
print(f"Mjerenja su izvršena na {broj_osoba} osoba.")

plt.scatter(data[:, 1], data[:, 2], color='blue')
plt.title('Odnos visine i mase')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.show()

plt.scatter(data[::50, 1], data[::50, 2], color='red')
plt.title('Odnos visine i mase (svaka pedeseta osoba)')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.show()

min_visina = np.min(data[:, 1])
max_visina = np.max(data[:, 1])
mean_visina = np.mean(data[:, 1])

print(f"Minimalna visina: {min_visina} cm")
print(f"Maksimalna visina: {max_visina} cm")
print(f"Srednja visina: {mean_visina} cm")

muskarci = data[data[:, 0] == 1]
min_visina_muskarci = np.min(muskarci[:, 1])
max_visina_muskarci = np.max(muskarci[:, 1])
mean_visina_muskarci = np.mean(muskarci[:, 1])

zene = data[data[:, 0] == 0]
min_visina_zene = np.min(zene[:, 1])
max_visina_zene = np.max(zene[:, 1])
mean_visina_zene = np.mean(zene[:, 1])

print(f"\nMuškarci:")
print(f"Minimalna visina: {min_visina_muskarci} cm")
print(f"Maksimalna visina: {max_visina_muskarci} cm")
print(f"Srednja visina: {mean_visina_muskarci} cm")

print(f"\nŽene:")
print(f"Minimalna visina: {min_visina_zene} cm")
print(f"Maksimalna visina: {max_visina_zene} cm")
print(f"Srednja visina: {mean_visina_zene} cm")
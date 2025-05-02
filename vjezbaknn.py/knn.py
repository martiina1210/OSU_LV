from sklearn.neighbors import KNeighborsClassifier

# X = značajke, y = ciljne klase
X = [[150, 0], [170, 0], [140, 1], [130, 1]]
y = ['Jabuka', 'Jabuka', 'Višnja', 'Višnja']

# Model s K = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Nova točka
nova = [[145, 0]]
predikcija = model.predict(nova)
print(predikcija)  # Izlaz: ['Visnja']

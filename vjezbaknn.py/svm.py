from sklearn import svm

# Primjer podataka (X: značajke, y: klase)
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]

# Linearni SVM
model = svm.SVC(kernel='linear', C=1000)
model.fit(X, y)

# Predikcija
nova = [[0.8, 0.8]]
print(model.predict(nova))  # Izlaz: [1]
print("Koeficijenti w:", model.coef_)
print("Bias b:", model.intercept_)

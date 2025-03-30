import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


data, labels = make_classification(n_samples=500, n_features=2, n_classes=2, 
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', label='Train')
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='x', label='Test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Prikaz podataka')
plt.show()


model = LogisticRegression()
model.fit(X_train, y_train)


w1, w2 = model.coef_[0]
bias = model.intercept_[0]
print(f'Model parametri: w1={w1}, w2={w2}, bias={bias}')


x_values = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
y_values = -(w1 / w2) * x_values - (bias / w2)

plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm')
plt.plot(x_values, y_values, 'k--', label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Granica odlučivanja')
plt.show()


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Matrica zabune:')
print(cm)
print(classification_report(y_test, y_pred))


plt.figure(figsize=(8,6))
correct = (y_test == y_pred)
plt.scatter(X_test[correct,0], X_test[correct,1], c='green', label='Correctly Classified')
plt.scatter(X_test[~correct,0], X_test[~correct,1], c='black', label='Misclassified')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Prikaz klasifikacije testnog skupa')
plt.show()

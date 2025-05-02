import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("lv6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()



# KNN model s K=5
KNN_model = KNeighborsClassifier(n_neighbors=5, p=2)
KNN_model.fit(X_train_n, y_train)

# Evaluacija KNN modela
y_train_p_knn = KNN_model.predict(X_train_n)
y_test_p_knn = KNN_model.predict(X_test_n)

print("KNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_knn))))

# granica odluke pomocu KNN (K=5)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend(loc='upper left')
plt.title("KNN (K=5) - Točnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_knn))))
plt.tight_layout()
plt.show()

# 2. Analiza granica odluke za različite vrijednosti K
for k in [1, 5, 100]:
    KNN_model_k = KNeighborsClassifier(n_neighbors=k)
    KNN_model_k.fit(X_train_n, y_train)

    # granica odluke pomocu KNN za razlicite vrijednosti K
    plot_decision_regions(X_train_n, y_train, classifier=KNN_model_k)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend(loc='upper left')
    plt.title(f"KNN (K={k}) - Točnost: " + "{:0.3f}".format(accuracy_score(y_train, KNN_model_k.predict(X_train_n))))
    plt.tight_layout()
    plt.show()

    # Ispis točnosti za treniranje i testiranje za različite K
    print(f"KNN (K={k}): ")
    print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, KNN_model_k.predict(X_train_n)))))
    print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, KNN_model_k.predict(X_test_n)))))





# Definiramo raspon vrijednosti K za testiranje
param_grid = {'n_neighbors': np.arange(1, 21)}

# KNN model
knn = KNeighborsClassifier()

# GridSearchCV sa 10-strukom unakrsnom validacijom
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)

# Izvodimo unakrsnu validaciju na skupu za treniranje
grid_search.fit(X_train_n, y_train)

# Optimalni hiperparametar (K)
best_k = grid_search.best_params_['n_neighbors']
print(f"Optimalna vrijednost K: {best_k}")

# Ispisujemo najbolju točnost (najbolji rezultat unakrsne validacije)
print(f"Najbolja točnost na skupu za treniranje: {grid_search.best_score_:.3f}")

# Provodimo predikciju koristeći najbolji model
best_knn_model = grid_search.best_estimator_

# Evaluacija na skupu za treniranje i testiranje
y_train_p_best_knn = best_knn_model.predict(X_train_n)
y_test_p_best_knn = best_knn_model.predict(X_test_n)

print("KNN (optimalni K): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_best_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_best_knn))))

# Granica odluke za najbolji KNN model
plot_decision_regions(X_train_n, y_train, classifier=best_knn_model)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend(loc='upper left')
plt.title(f"KNN (Optimalni K={best_k}) - Točnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_best_knn))))
plt.tight_layout()
plt.show()


#SVM


# Funkcija za primjenu SVM modela
def apply_svm(kernel_type='rbf', C=1, gamma='scale', degree=3):
    # Kreiramo SVM model sa zadanim parametrima
    svm_model = SVC(kernel=kernel_type, C=C, gamma=gamma, degree=degree)
    svm_model.fit(X_train_n, y_train)
    
    # Predikcija na trening skupu i test skupu
    y_train_pred = svm_model.predict(X_train_n)
    y_test_pred = svm_model.predict(X_test_n)
    
    # Ispis točnosti
    print(f"SVM Model with {kernel_type} Kernel, C={C}, Gamma={gamma}:")
    print(f"Accuracy on Train Set: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"Accuracy on Test Set: {accuracy_score(y_test, y_test_pred):.3f}")
    
    # Prikazivanje granice odluke
    plot_decision_regions(X_train_n, y_train, classifier=svm_model)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend(loc='upper left')
    plt.title(f"Decision Boundary: {kernel_type} Kernel")
    plt.tight_layout()
    plt.show()

# 1. Primjena SVM s RBF kernelom i različitim vrijednostima C i gamma
apply_svm(kernel_type='rbf', C=1, gamma=1)
apply_svm(kernel_type='rbf', C=10, gamma=1)
apply_svm(kernel_type='rbf', C=1, gamma=0.1)
apply_svm(kernel_type='rbf', C=10, gamma=0.1)

# 2. Primjena SVM s Linearnim kernelom
apply_svm(kernel_type='linear', C=1, gamma=1)
apply_svm(kernel_type='linear', C=10, gamma=1)

# 3. Primjena SVM s Polinomskim kernelom
apply_svm(kernel_type='poly', C=1, gamma='scale')
apply_svm(kernel_type='poly', C=10, gamma='scale')


#ZADNJI


# Definiramo raspon vrijednosti za C i gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# Kreiramo SVM model sa RBF kernelom
svm_model = SVC(kernel='rbf')

# GridSearchCV za pronalazak najboljih hiperparametara
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=1)

# Treniranje modela
grid_search.fit(X_train_n, y_train)

# Optimalni hiperparametri
best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
print(f"Optimalna vrijednost C: {best_C}")
print(f"Optimalna vrijednost gamma: {best_gamma}")

# Ispisujemo najbolju točnost (najbolji rezultat unakrsne validacije)
print(f"Najbolja točnost na skupu za treniranje (kroz unakrsnu validaciju): {grid_search.best_score_:.3f}")

# Provodimo evaluaciju na testnom skupu s najboljim modelom
best_svm_model = grid_search.best_estimator_

y_train_pred = best_svm_model.predict(X_train_n)
y_test_pred = best_svm_model.predict(X_test_n)

print(f"SVM sa najboljim parametrima (C={best_C}, gamma={best_gamma}):")
print(f"Točnost na trening skupu: {accuracy_score(y_train, y_train_pred):.3f}")
print(f"Točnost na test skupu: {accuracy_score(y_test, y_test_pred):.3f}")

# Prikazivanje granice odluke za najbolji model
plot_decision_regions(X_train_n, y_train, classifier=best_svm_model)
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend(loc='upper left')
plt.title(f"SVM (Optimalni C={best_C}, gamma={best_gamma}) - Točnost: {accuracy_score(y_train, y_train_pred):.3f}")
plt.tight_layout()
plt.show()


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
#Kada radiš s modelima strojno učenja (npr. KNN, SVM, logistička regresija) 
# iz scikit-learn-a, većina tih modela zahtijeva da podaci budu u NumPy formatu 
# (tj. kao np.array). Pandas DataFrame može sadržavati podatke u različitim formatima 
# (tekst, brojevi, itd.), ali modeli u scikit-learn obično očekuju jednodimenzionalne 
# ili dvodimenzionalne nizove brojeva. Zato je potrebno pretvoriti podatke iz Pandas 
# DataFrame-a u NumPy nizove.

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)
#stratify je ključan parametar. Kada se koristi, osigurava da raspodjela klasa u trening skupu 
# i testnom skupu bude proporcionalna onoj koja postoji u cijelom skupu podataka.

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

#fit_transform(X_train):
#fit() metoda izračunava srednju vrijednost (mean) i standardnu devijaciju (standard deviation) za svaku značajku u trening skupu (X_train).
#transform() primjenjuje standardizaciju na trening skup podataka koristeći ove izračunate vrijednosti.
#Rezultat je da su sve značajke u X_train standardizirane: svaka značajka će imati srednju vrijednost 0 i standardnu devijaciju 1.
#Dakle, X_train_n sadrži standardizirane vrijednosti za trening podatke.
#transform(X_test):
#Kada radimo s testnim podacima (X_test), važno je koristiti iste statistike (srednju vrijednost i standardnu devijaciju) koje smo izračunali iz trening skupa.
#fit_transform() ne bi trebalo biti primijenjeno na testne podatke jer želimo da testni podaci budu transformirani na temelju statistika trening skupa.
#transform() samo koristi srednju vrijednost i standardnu devijaciju iz trening skupa za skaliranje testnog skupa (X_test).
#To znači da X_test_n sadrži skalirane testne podatke na temelju statistika iz X_train.

#Ako su značajke poput dobi (npr. između 20 i 70) i plaće (npr. između 10,000 i 200,000) 
# na različitim skalama, algoritmi koji koriste udaljenost (npr. KNN, SVM) mogu biti 
# previše osjetljivi na veće značajke (kao što je plaća), zanemarujući značajke s manjim 
# rasponima (kao što je dob).

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
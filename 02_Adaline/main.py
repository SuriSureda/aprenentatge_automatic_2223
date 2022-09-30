import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  make_classification
from Adaline import Adaline
from sklearn.preprocessing import Normalizer

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                           random_state=8)
y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


# TODO: Normalitzar les dades
normalizedX = Normalizer().fit_transform(X)
# TODO: Entrenar usant l'algorisme de Batch gradient
adaline = Adaline(eta=0.00000001, n_iter=100)
# TODO: Mostrar els resultats
adaline.fit(X, y)

y_prediction = adaline.predict(X)

###  Mostram els resultats
plt.figure(1)
# Dibuixam el núvol de punts (el parametre c indica que colorejam segons la classe)
plt.scatter(X[:, 0], X[:, 1], c=y)

# Dibuixem la recta. Usam l'equació punt-pendent
m = -adaline.w_[1] / adaline.w_[2]
origen = (0, -adaline.w_[0] / adaline.w_[2])
plt.axline(xy1=origen, slope=m)
plt.savefig('output.png')
plt.show()

### Extra: Dibuixam el nombre d'errors en cada iteracio de l'algorisme
# plt.figure(2)
# plt.plot(adaline.errors_, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Number of errors')
# plt.show()
# plt.savefig('output-errors.png')

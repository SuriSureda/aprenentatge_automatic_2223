from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  make_classification
from Adaline import Adaline
from sklearn.preprocessing import StandardScaler

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                           random_state=8)
y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


# TODO: Normalitzar les dades
scaler = StandardScaler()
normalizedX = scaler.fit_transform(X)
# TODO: Entrenar usant l'algorisme de Batch gradient

n_iter = 500

adaline = Adaline(eta=0.0001, n_iter=n_iter)
# TODO: Mostrar els resultats
adaline.fit(normalizedX, y)

###  Mostram els resultats
plt.figure(1)
# Dibuixam el núvol de punts (el parametre c indica que colorejam segons la classe)
plt.scatter(normalizedX[:, 0], normalizedX[:, 1], c=y)

# Dibuixem la recta. Usam l'equació punt-pendent
m = -adaline.w_[1] / adaline.w_[2]
origen = (0, -adaline.w_[0] / adaline.w_[2])
plt.axline(xy1=origen, slope=m)
# plt.savefig('output.png')
plt.show()

### Extra: Dibuixam el nombre d'errors en cada iteracio de l'algorisme
plt.figure(2)
plt.plot(range(1, n_iter + 1), adaline.cost_)
plt.xlabel('Epochs')
plt.ylabel('Sum of squared error')
# plt.savefig('output-errors.png')
plt.show()

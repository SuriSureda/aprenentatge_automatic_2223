import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  make_classification
from Adaline import Adaline

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.5,
                           random_state=8)
y[y == 0] = -1  # La nostra implementació esta pensada per tenir les classes 1 i -1.


# TODO: Normalitzar les dades
# TODO: Entrenar usant l'algorisme de Batch gradient
# TODO: Mostrar els resultats

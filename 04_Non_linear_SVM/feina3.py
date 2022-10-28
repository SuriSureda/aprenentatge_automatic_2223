# Generació del conjunt de mostres
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = make_classification(n_samples=100, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=2, class_sep=0.5,
                           random_state=9)
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y,  shuffle=True, random_state=123, test_size=0.25)
# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)
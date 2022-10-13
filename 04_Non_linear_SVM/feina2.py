from dis import dis
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from scipy.spatial import distance_matrix
from numpy.linalg import matrix_power

# Generació del conjunt de mostres
X, y = make_classification(n_samples=100, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=2, class_sep=0.5,
                           random_state=9)
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y,  shuffle=True, random_state=123, test_size=0.25)
# Els dos algorismes es beneficien d'estandaritzar les dades
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

gamma = 0.01
## SVC defined linear kernel
svm = SVC(kernel='rbf', C=1, gamma=gamma)
svm.fit(X_train, y_train);
X_test_transformed = scaler.transform(X_test)

## SVC with custom linear kernel
def kernel_gauss(x1, x2, gamma = gamma):
	dist_mat = distance_matrix(x1, x2)
	dist_mat = -gamma*dist_mat**2
	return np.exp(dist_mat)
	
svmCustom = SVC(kernel=kernel_gauss, C=1)
svmCustom.fit(X_train, y_train)

svmCustom.predict(X_test) == svm.predict(X_test)

# RESULTS
y_predicted = svm.predict(X_test_transformed)
y_customPredicted = svmCustom.predict(X_test_transformed)

print("SVM Predict == SVM CUSTOM Predict : ", np.array_equal(y_predicted, y_customPredicted))

# ERRORS BY SVM
differences = (y_predicted - y_test)
errors = np.count_nonzero(differences)
print(f'[SVM] Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')

differences = (y_customPredicted - y_test)
errors = np.count_nonzero(differences)
print(f'[SVM CUSTOM]Rati d\'acerts en el bloc de predicció: {(len(y_predicted)-errors)/len(y_predicted)}')
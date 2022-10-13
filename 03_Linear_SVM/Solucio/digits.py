import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# En carregar un dataset de scikit obtenim un diccionari
digits = load_digits()
full_data = digits.data
y = digits.target

# Visualitzem un exemple
imatge = np.reshape(full_data[0, :], (8, 8))
plt.imshow(imatge)
plt.title(y[0])
plt.show()

# Seleccionam 4 nombres
indexos_0_4 = np.where((y >= 0) & (y < 4))
X = full_data[indexos_0_4]
y = y[indexos_0_4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

svm = SVC(C=0.1, kernel='linear')
svm.fit(X_transformed, y_train)
y_predicted = svm.predict(X_test_transformed)

# Classification report
print(classification_report(y_test, y_predicted, target_names=["0","1","2","3"]))
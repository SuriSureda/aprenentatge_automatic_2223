# https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb#scrollTo=alow2xGBShAc
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


digits = datasets.load_digits()
plot_digits(digits.data[:100, :])
print(digits.data.shape)

pca = PCA().fit(digits.data)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

pca = PCA(0.99, whiten=True)
data = pca.fit_transform(digits.data)
print(data.shape)

plt.figure()
n_components = np.arange(50, 110, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0)
          for n in n_components]
aics = [model.fit(data).bic(data) for model in models]
plt.plot(n_components, aics)


gmm = GaussianMixture(70, covariance_type='full', random_state=0)
gmm.fit(data)
print(gmm.converged_)

data_new, _ = gmm.sample(100)
print(data_new.shape)

digits_new = pca.inverse_transform(data_new)
plot_digits(digits_new)
plt.show()

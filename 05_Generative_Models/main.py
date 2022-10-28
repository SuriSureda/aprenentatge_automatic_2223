# Andreu Sureda

import matplotlib.pyplot as plt
from pyparsing import col
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

def plot_digits(data):
    fig, ax = plt.subplots(10, 10, figsize=(10, 10),
                           subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)


digits = datasets.load_digits()

X = digits.data

pca = PCA()
pca_components = pca.fit_transform(X)
pca_variance = pca.explained_variance_ratio_

acc_variance = np.cumsum(pca_variance)
plt.plot(acc_variance)

# SELECCIÓ MILLOR NOMBRE DE COMPONENTS PCA
def get_best_n_components(variances, min = 0.96, diff = 0.005):
    n_components= 1
    acc_var = variances[0]
    prev_acc = 0
    while (acc_var < min or acc_var - prev_acc > diff): 
        prev_acc = acc_var
        acc_var = variances[n_components]
        n_components +=1
    return (n_components, acc_var)

best_n, variance = get_best_n_components(acc_variance)
plt.axvline(x = best_n, color='r')
plt.axhline(y = 0.96, color = 'b', linestyle='--', label='Minimum variance')
plt.axhline(y = variance, color = 'g', label='Actual variance')
plt.legend()
# plt.savefig("accumulate_variance.png")
# plt.clf()
plt.show()

# PCA AMB MILLOR N
pca = PCA(n_components=best_n)
pca_components = pca.fit_transform(X)

# MIXTURA DE GAUSSIANES
def get_gaussian_bic_list(values, max_rang):
    _list = []
    for i in range(1, max_rang+1):
        gm = GaussianMixture(n_components=i, random_state=1)
        gm.fit(values)
        bic = gm.bic(values)
        _list.append(bic)
    return _list


gm_bic_list = get_gaussian_bic_list(pca_components, 10)
best_gm_n_components = np.argmin(gm_bic_list) + 1
plt.axvline(x = best_gm_n_components, color='r')
plt.plot(list(range(1,11)), gm_bic_list)
# plt.savefig("gaussian_bic.png")
# plt.clf()
plt.show()

# GENERACIÓ
gm = GaussianMixture(n_components= best_gm_n_components, random_state=1)
gm.fit(pca_components)

samples = gm.sample(100)
generated_samples = pca.inverse_transform(samples[0])
plot_digits(generated_samples[:100, :])
plt.show()
# plt.savefig("generated_numbers.png")
# plt.clf()

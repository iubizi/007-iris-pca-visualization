####################
# load iris
####################

from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

####################
# pca
####################

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x)

# display ratio
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# pca dimension reduction
x_pca = pca.transform(x)

####################
# visualization
####################

import matplotlib.pyplot as plt

# different class have different color
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y) # marker='.'
plt.show()

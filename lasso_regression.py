from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import mglearn 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression

## For lasso: some coefficients could be exactly zero
## Similarly to Ridge, the Lasso also has a regularization parameter, alpha, that controls
# how strongly coefficients are pushed toward zero. I

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))


## the score here is very low which indicates that we are underfitting 
## and we also find out that lasso is only using four features!


## in the example above, we are using alpha = 1. We could try to decrease alpha so that lsso could use more features

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score after adjusting alpha: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score after adjusting alpha: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))



plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso00001.coef_, '^', label="Lasso alpha=0.01")

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()


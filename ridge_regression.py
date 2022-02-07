from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import mglearn 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge().fit(X_train, y_train)


print("Training set score -- Linear Model: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score -- Linear Model: {:.2f}".format(lr.score(X_test, y_test)))
print("Training set score -- Ridge Model: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score -- Ridge Model: {:.2f}".format(ridge.score(X_test, y_test)))


# As you can see, the training set score of Ridge is lower than for LinearRegression,
# while the test set score is higher. This is consistent with our expectation. With linear
# regression, we were overfitting our data. Ridge is a more restricted model, so we are
# less likely to overfit.

# alpha is also a parameter that we could use
# this parameter makes the model less strict about coefficients 


ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score -- alpha 10: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score  --alpha 10 : {:.2f}".format(ridge10.score(X_test, y_test)))
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score -- alpha 0.1: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score --alpha 0.1: {:.2f}".format(ridge01.score(X_test, y_test)))


plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()

## Here, alpha=0.1 seems to be working well. We could try decreasing alpha even more
# to improve generalization.
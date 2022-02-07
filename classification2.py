from sklearn.model_selection import train_test_split
import mglearn 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


X, y = mglearn.datasets.make_forge()

print("shape of X", X.shape)
print("shape of y", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("shape of X train", X_train.shape)
print("shape of y train", y_train.shape)
print("shape of X test", X_test.shape)
print("shape of Y_test", y_test.shape)


clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)


## For each data point in the test set, this computes its nearest neighbors in the training set and finds the
## most common class among these

print("Test set predictions: {}".format(clf.predict(X_test)))


print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

#We see that our model is about 86% accurate, meaning the model predicted the class correctly for 86% of the samples in the test dataset.




# 3 is just an arbitrary number that we set above.
# what if we change 3 to other numbers? 

# here, we set nearest neighbhor to 1, 3, and 9

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
 # the fit method returns the object self, so we can instantiate
 # and fit in one line
 clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
 mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
 mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
 ax.set_title("{} neighbor(s)".format(n_neighbors))
 ax.set_xlabel("feature 0")
 ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

plt.show()


#A smoother boundary corresponds to a simpler model. In other words, using few neighbors corresponds to high model com‐
# plexity (as shown on the right side of Figure 2-1), and using many neighbors corre‐
# sponds to low model complexity
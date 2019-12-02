from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import pandas as pd

dataset = pd.read_csv("diabetes.csv")
train_set, test_set = train_test_split(dataset, test_size=.33)

# Separate labels from the rest of the dataset
train_set_labels = train_set["class"].copy()
train_set = train_set.drop("class", axis=1)

test_set_labels = test_set["class"].copy()
test_set = test_set.drop("class", axis=1)

# Apply a scaler
from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

X_train = train_set_scaled
Y_train = train_set_labels
X_test = test_set_scaled
Y_test = test_set_labels

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, Y_train)

predictions = knn3.predict(X_test)
print("knn3", accuracy_score(Y_test, predictions))

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, Y_train)

predictions = knn5.predict(X_test)
print("knn5", accuracy_score(Y_test, predictions))

knn11 = KNeighborsClassifier(n_neighbors=11)
knn11.fit(X_train, Y_train)

predictions = knn11.predict(X_test)
print("knn11", accuracy_score(Y_test, predictions))

treeDec = tree.DecisionTreeClassifier(random_state=0)
treeDec.fit(X_train, Y_train)

predictions=treeDec.predict(X_test)
print("Drzewa decyzyjne", accuracy_score(Y_test, predictions))

nb = tree.DecisionTreeClassifier(random_state=0)
nb.fit(X_train, Y_train)

predictions=nb.predict(X_test)
print("Naiwny bayesowski", accuracy_score(Y_test, predictions))
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.tree.export import export_text
from sklearn.metrics import confusion_matrix
from sklearn import svm

import numpy as np
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
x = iris.data
y = iris.target

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)

# Use decistion tree algorithm
classifier = tree.DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)

r = export_text(classifier, feature_names=iris['feature_names'])
print(r)

import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

# tree.plot_tree(classifier)
# tree.export_graphviz(classifier)
# tree.plot_tree(classifier.fit(x, y))

predictions=classifier.predict(x_test)
print(accuracy_score(y_test, predictions))
print(iris.target_names)
cm = confusion_matrix(y_test, predictions)

print(cm)

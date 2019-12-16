import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

print(df.head())

rows, cols = df.shape
print(rows)
print(cols)

label_names = df['Species'].unique()
print(label_names)

index_and_label = list(enumerate(label_names))
print(index_and_label)

label_to_index = dict((label, index) for index, label in index_and_label)
print(label_to_index)

df = df.replace(label_to_index)

df = df.sample(frac=1)

train_data = df.iloc[:120, :]
test_data = df.iloc[120:, :]

x_train = train_data.iloc[:120, 1:-1]
y_train = train_data.iloc[:120, -1:]

x_test = test_data.iloc[120:, 1:-1]
y_test = test_data.iloc[120:, -1:]

def scale_column(train_data, test_data, column):
    min_value = train_data[column].min()
    max_value = train_data[column].max()
    train_data[column] = (train_data[column] - min_value)/(max_value - min_value)
    test_data[column] = (test_data[column] - min_value)/(max_value - min_value)

scale_column(x_train, x_test, 'SepalLengthCm')
scale_column(x_train, x_test, 'SepalWidthCm')
scale_column(x_train, x_test, 'PetalLengthCm')
scale_column(x_train, x_test, 'PetalWidthCm')

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=[4]))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
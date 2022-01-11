# from __future__ import print_function, division
# from future.utils import iteritems
# from builtins import range, input
# import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat', sep='\t', header=None)

df.head()

print(df.info())

data = df[[0, 1, 2, 3, 4]].values

print(df.values)

print(data)

target = df[5].values


X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.33)


model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))

predictions = model.predict(X_test)

print(predictions)

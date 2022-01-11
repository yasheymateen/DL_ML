from __future__ import print_function, division
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:10:14 2022

@author: yashe
"""
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from future.utils import iteritems
# pip install -U future
from builtins import range, input

import numpy as np

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

model = RandomForestClassifier()
model.fit(X_train, y_train)


model.score(X_train, y_train)

model.score(X_test, y_test)

predictions = model.predict(X_test)

print(predictions)

scaler = StandardScaler()

X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

model = MLPClassifier(max_iter=500)

model.fit(X_train2, y_train)

print(model.score(X_train2, y_train))

print(model.score(X_test2, y_test))


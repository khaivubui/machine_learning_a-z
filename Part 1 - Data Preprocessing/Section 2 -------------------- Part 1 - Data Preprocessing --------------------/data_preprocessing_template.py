# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
X[:, 1:3] = Imputer().fit(X[:, 1:3]).transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
X = OneHotEncoder(categorical_features = [0]).fit_transform(X).toarray()

y = LabelEncoder().fit_transform(y)
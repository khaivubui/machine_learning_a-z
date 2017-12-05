# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
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

# Splitting dataset to Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
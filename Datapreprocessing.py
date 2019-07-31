#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing the data set

file_path = 'D:\Hacking\DataSet for ML\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data.csv'
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :-1].values # Making Matrices
Y = dataset.iloc[:, 3].values

#Filling missing data with mean of the column

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotecoder = OneHotEncoder(categorical_features = [0])
X = onehotecoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#splitong my dataset into two sets training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train, X_test, Y_train, Y_test)
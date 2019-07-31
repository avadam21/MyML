#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing the data set

file_path = 'D:\Hacking\DataSet for ML\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\Data.csv'
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, :].values # Making Matrices
Y = dataset.iloc[:, ].values

#splitong my dataset into two sets training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X_train, X_test, Y_train, Y_test)
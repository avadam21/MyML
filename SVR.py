#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing dataset

filepath = 'D:\Hacking\DataSet for ML\SVR\Svr.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values
print(Y)

#Fitting SVR to the dataset

regressor = SVR(kernel= 'rbf')
regressor.fit(X, Y)

#Feature Scalling

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
print(X)

#Predicting a new result

Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[ 6.5]]))))
print(Y_pred)

#Visualizing the SVR Result

mlp.scatter(X, Y, color = 'red')
mlp.plot(X, regressor.predict(X), color = 'blue')
mlp.title('Truth or Bluff (SVR)')
mlp.xlabel('Position Level')
mlp.ylabel('Salary')
mlp.show()

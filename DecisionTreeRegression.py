#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing dataset

file_path = 'D:\Hacking\DataSet for ML\Decision_Tree_Regression\Position_Salaries.csv'
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values
print(Y)

#Fitting decision tree regression to the dataset

regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X, Y)
"""
#Feature Scalling

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
print(X)
"""
#Predicting a new result

Y_pred = regressor.predict([[6.5]])
print(Y_pred)

#Visualizing the decision tree regression Result

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)),1)
mlp.scatter(X, Y, color = 'red')
mlp.plot(X_grid, regressor.predict(X_grid), color = 'blue')
mlp.title('Truth or Bluff (Decision Tree Regression)')
mlp.xlabel('Position Level')
mlp.ylabel('Salary')
mlp.show()


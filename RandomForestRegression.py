#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.ensemble import RandomForestRegressor

#Importing dataset

file_path = 'D:\Hacking\DataSet for ML\Random_Forest_Regression\Position_Salaries.csv'
dataset = pd.read_csv(file_path)
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:].values
print(Y)

"""
#Feature Scalling

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)
print(X)
"""

#Fitting decision Random Forest regression to the dataset

regressor = RandomForestRegressor(n_estimators= 300, random_state= 0)
regressor.fit(X, Y)


#Predicting a new result

Y_pred = regressor.predict([[6.5]])
print(Y_pred)

#Visualizing the Random Forest regression Result

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)),1)
mlp.scatter(X, Y, color = 'red')
mlp.plot(X_grid, regressor.predict(X_grid), color = 'blue')
mlp.title('Truth or Bluff (Decision Tree Regression)')
mlp.xlabel('Position Level')
mlp.ylabel('Salary')
mlp.show()


#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Importing dataset

filepath = 'D:\Hacking\DataSet for ML\Polynomial_Regression\Position_Salaries.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Fitting Linear Regression Model to the dataset

linRegression = LinearRegression()
linRegression.fit(X, Y)

#Fitting Polynomial Regression model to the dataset

polyRegression = PolynomialFeatures(degree= 4)
X_poly = polyRegression.fit_transform(X)
linRegression_2 = LinearRegression()
linRegression_2.fit(X_poly, Y)

#Visualizing the Linear Regression Result

mlp.scatter(X, Y, color = 'red')
mlp.plot(X, linRegression.predict(X), color = 'blue')
mlp.title('Truth or Bluff (Linear Regression)')
mlp.xlabel('Position Level')
mlp.ylabel('Salary')
mlp.show()

#Visualizing the Polynomial Regression Result

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
mlp.scatter(X, Y, color = 'red')
mlp.plot(X_grid, linRegression_2.predict(polyRegression.fit_transform(X_grid)), color = 'blue')
mlp.title('Truth or Bluff (Linear Regression)')
mlp.xlabel('Position Level')
mlp.ylabel('Salary')
mlp.show()

#Predicting a new result using Linear Regression

print(linRegression.predict([[6.5]]))

#Predicting a new result using Polynomial Regression

print(linRegression_2.predict(polyRegression.fit_transform([[6.5]])))

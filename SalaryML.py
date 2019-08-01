#Importing the libraries

import numpy as np
import matplotlib.pyplot as mlp
import pandas as ps
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Importing the dataset

file_path = "D:\Hacking\DataSet for ML\Simple_Linear_Regression\Salary_Data.csv"
dataset = ps.read_csv(file_path)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Linear Regression to the dataset

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting Test Set data

Y_pred = regressor.predict(X_test)

#Plotting of the result set

mlp.scatter(X_test, Y_test, color = 'red')
mlp.plot(X_train, regressor.predict(X_train), color = 'violet')
mlp.title('Salary vs Experience')
mlp.xlabel('Experience')
mlp.ylabel('Salary')
mlp.show()


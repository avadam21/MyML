#Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as mlp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#Importing dataset

filepath = 'D:\Hacking\DataSet for ML\Multiple_Linear_Regression\Ab50_Startups.csv'
dataset = pd.read_csv(filepath)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4]

#Encoading the categorical data

labelX = LabelEncoder()
X[:, 3] = labelX.fit_transform(X[:, 3])
one = OneHotEncoder(categorical_features= [3])
X = one.fit_transform(X).toarray()

#Avoiding the dummy variable trap

X = X[:, 1:]

#Sppliting the dataset into test and train

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)

#Fitting multiple linear regression model to the training set

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predict the test set result

Y_pred = regressor.predict(X_test)

#Building thw optimal model using Backward elimination

X = np.append(arr= np.ones((50, 1)).astype(int), values= X, axis= 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5] ]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [0, 1, 3, 4, 5] ]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [0, 3, 4, 5] ]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [0, 3, 5] ]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
X_opt = X[:, [0, 3] ]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

#Polynomial Regression

#Data Processing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa

#Importing the Dataset
dataset = pa.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #X is Matrix and Y is a vector
y = dataset.iloc[:, 2].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#Visualising the Linear Regression Results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#For Linear regression model we always get a straight line

#Visualisng polynomial regression results
'''
##plt.scatter(x, y, color = 'red')
##plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
##plt.title('Truth or Bluff (Polynomial Regression)')
##plt.xlabel('Position Level')
##plt.ylabel('Salary')
##plt.show() '''
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

Required_Salary_of_new_employee = [[6.5],[0]]
#Predicting a new result with linear regression
#LinearRegression.predict(lin_reg(6.5).reshape(1,1))
lin_reg.predict(Required_Salary_of_new_employee)
#predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(Required_Salary_of_new_employee))




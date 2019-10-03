#Randon Forest Regression

#Regression Template

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa

#Importing the Dataset
dataset = pa.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #X is Matrix and Y is a vector
y = dataset.iloc[:, 2].values

'''#Splitting the dataset into Training Set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
''''''For Training Set we need to fit and Transform where as for test set we
only need to transform, because it is already fitted to training set''''''
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''
#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)


#predicting a new result with regression
Required_Salary_of_new_employee = [[6.5],[0]]
y_pred = regressor.predict(Required_Salary_of_new_employee)

#Visualisng Random Forest Regression results (For Higher Resolution and Smoother Curve)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#SVR
#support vector machine
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pa

#Importing the Dataset
dataset = pa.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #X is Matrix and Y is a vector
y = dataset.iloc[:, 2:3].values

'''#Splitting the dataset into Training Set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)'''

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

#predicting a new result with SVR
#Required_Salary_of_new_employee = [sc_x.fit_transform([6.5],[0])]
#Required_Salary_of_new_employee = sc_x.fit_transform(np.array[[6.5]])
#y_pred = regressor.predict(Required_Salary_of_new_employee)
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5],[0]])))
#Visualisng SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualisng SVR results (For Higher Resolution and Smoother Curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
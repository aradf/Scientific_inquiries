# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# ski-kit-learn library has split data set to training and test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, 
                                                     y, 
                                                     test_size = 1/3, 
                                                     random_state = 0)

# Training the Simple Linear Regression model on the Training set
# ski-kit-learn library has linear regresion modules.
# the fit method of the LinearRegression trains the model.
from sklearn.linear_model import LinearRegression
regressor_model = LinearRegression()
regressor_model.fit( X_train, 
                     y_train )

# Predicting the Test set results
y_predicted = regressor_model.predict ( X_test )

# Visualising the Training set results
plt.scatter( X_train, 
             y_train, 
             color = 'red' )

plt.plot( X_train, 
          regressor_model.predict( X_train ), 
          color = 'blue' )

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter( X_test, 
             y_test, 
             color = 'red' )

plt.plot( X_train, 
          regressor_model.predict( X_train ), 
          color = 'blue')

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print (regressor_model.coef_, regressor_model.intercept_)
print ( regressor_model.predict( np.array([[12.0]]) ) )

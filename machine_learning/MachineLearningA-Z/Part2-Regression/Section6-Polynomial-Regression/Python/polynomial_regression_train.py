# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit( X, 
                       y )

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures( degree = 4 )
X_polynomial = polynomial_regression.fit_transform( X )
linear_regression_2 = LinearRegression()
linear_regression_2.fit( X_polynomial, 
                         y)

# Visualising the Linear Regression results
plt.scatter( X, 
             y, 
             color = 'red')
plt.plot( X, 
          linear_regression.predict( X ), 
          color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter( X, 
             y, 
             color = 'red')
plt.plot( X, 
          linear_regression_2.predict( polynomial_regression.fit_transform( X ) ), 
          color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange( min( X ), 
                    max( X ), 
                    0.1)

X_grid = X_grid.reshape( ( len( X_grid ), 1 ) )
plt.scatter( X, 
             y, 
             color = 'red')
plt.plot( X_grid, 
          linear_regression_2.predict( polynomial_regression.fit_transform( X_grid ) ), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print ( linear_regression.predict([[6.5]]) )

# Predicting a new result with Polynomial Regression
print ( linear_regression_2.predict(polynomial_regression.fit_transform([[6.5]])) )
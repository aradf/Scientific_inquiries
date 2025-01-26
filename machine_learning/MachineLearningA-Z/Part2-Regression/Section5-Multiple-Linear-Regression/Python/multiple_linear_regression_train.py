# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# integer base indexing (pd.iloc[row index, col index])
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
# sci-kit-learn library compose and ColumnTransformer
# sci-kit-learn library preprocessing and OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column_transfer = ColumnTransformer(transformers=[( 'encoder', 
                                       OneHotEncoder(), 
                                       [3])], remainder='passthrough')

X = np.array( column_transfer.fit_transform( X ) )
print( X )

# Splitting the dataset into the Training set and Test set
# sci-kit-learn library model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, 
                                                     y, 
                                                     test_size = 0.2, 
                                                     random_state = 0)

# Training the Multiple Linear Regression model on the Training set
# ski-kit-learn library has linear regression object.
from sklearn.linear_model import LinearRegression
regressor_model = LinearRegression()
regressor_model.fit( X_train, 
                     y_train )


# Predicting the Test set results
# reshape ( number of rows, number of columns)
# concatnate 1 == horizental and 0 == vertical
y_predicted = regressor_model.predict( X_test )
np.set_printoptions( precision = 2 )
print("Predicted, observed (acutal)")
print(np.concatenate(( y_predicted.reshape( len( y_predicted ), 1 ), 
                       y_test.reshape( len( y_test ), 1 ))
                       , 1 ) )

# building the optimal model using backward elemination.
# add a column of 1's in front of the X
import statsmodels.api as stats_models
X = np.append( arr = np.ones((50,1)).astype(int), 
               values = X, 
               axis = 1)

significance_level = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
X_optimal = X_optimal.astype(np.float64)

# ordinary least squared
regressor_modelOLS = stats_models.OLS( endog = y, exog = X_optimal ).fit()
print(regressor_modelOLS.summary())

print ( X_optimal )

# Question 1: How do I use my multiple linear regression model to make a single prediction, 
# for example, the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, 
# Marketing Spend = 300000 and State = California?
# X_single = [[0.0 1.0 0.0 160000 130000 300000]]
X_single = [[1.0, 0.0, 0.0, 160000.0, 130000.0, 300000.0]]
print( regressor_model.predict( X_single ) )
print( regressor_model.coef_)
print( regressor_model.intercept_)



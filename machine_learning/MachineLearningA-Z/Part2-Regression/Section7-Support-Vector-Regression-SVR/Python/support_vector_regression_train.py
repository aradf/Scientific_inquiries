# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
y = y.reshape( len(y),
               1 )
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
# sci-kit learn library preprocessing and standard Scaler (normalization and standardization)
# must apply feature scaling to both the dependent and independent variables, since the 
# indepdenent variables are 1 to 10 and the independents are much larger from 4500 to 100000.
standard_scalerX = StandardScaler()
standard_scalery = StandardScaler()
X = standard_scalerX.fit_transform(X)
y = standard_scalery.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset
# sci-kit learn library suport vector machine (svm) and support vector regression (svr)
# https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVR.html
# https://en.wikipedia.org/wiki/Radial_basis_function_kernel
# Radial basis function kernel (rbf)
# https://data-flair.training/blogs/svm-kernel-functions/
from sklearn.svm import SVR
regressor_modelSVR = SVR(kernel = 'rbf')
regressor_modelSVR.fit(X, y)

# Predicting a single new result
# some_object1 = standard_scalarX.transform (( [[ 6.5 ]])).reshape(-1,1)
# some object2 = regressor_modelSVR ( some_object1 )
# standard_scalery.inverse_transform ( some_object2 )
standard_scalery.inverse_transform( regressor_modelSVR.predict( standard_scalerX.transform([[6.5]]) ).reshape(-1,1) )

# Visualising the SVR results
plt.scatter( standard_scalerX.inverse_transform(X), 
             standard_scalery.inverse_transform(y), 
             color = 'red')
plt.plot( standard_scalerX.inverse_transform(X), 
          standard_scalery.inverse_transform(regressor_modelSVR.predict(X).reshape(-1,1)), 
          color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange( min( standard_scalerX.inverse_transform(X) ), 
                    max( standard_scalerX.inverse_transform(X) ), 
                    0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter( standard_scalerX.inverse_transform(X), 
             standard_scalery.inverse_transform(y), 
             color = 'red' )
plt.plot( X_grid, 
          standard_scalery.inverse_transform(regressor_modelSVR.predict(standard_scalerX.transform(X_grid)).reshape(-1,1)), 
          color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

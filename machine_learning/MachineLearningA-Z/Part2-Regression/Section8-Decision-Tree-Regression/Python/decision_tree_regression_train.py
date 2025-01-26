# Decision Tree Regression
# train a decision tree regression model.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
# ski-kit learned model tree and DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
regressor_decisionTree = DecisionTreeRegressor(random_state = 0)
regressor_decisionTree.fit( X, 
                            y)

# Predicting a new result
print ( regressor_decisionTree.predict([[6.5]]) ) 

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange( min(X), 
                    max(X), 
                    0.01 )

print ( X_grid.shape )
X_grid = X_grid.reshape(( len(X_grid), 
                          1 ))

print ( X_grid.shape )
plt.scatter( X, 
             y, 
             color = 'red')

plt.plot( X_grid, 
          regressor_decisionTree.predict(X_grid), 
          color = 'blue')

plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
print ("Hello")

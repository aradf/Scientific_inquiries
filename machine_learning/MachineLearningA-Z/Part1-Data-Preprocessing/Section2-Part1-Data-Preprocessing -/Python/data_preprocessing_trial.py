# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset, create matrix of features (independent variables ), dependent variables.
# locate index (iloc), [:, :-1] means all rows and all columns minues 1.
# data_set = pd.read_csv("iris.csv")
data_set = pd.read_csv("Data.csv")
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values
print(X)
print(y)

# Taking care of missing data, replace the missing data with average values
# sci - kit - learn (sklearn) 
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer( missing_values=np.nan, 
                                strategy='mean')

# all rows and columns 1 through 3
simple_imputer.fit( X[:, 1:3] )
X[:, 1:3] = simple_imputer.transform( X[:, 1:3])
print(X)

# Encoding categorical data
# Encoding the Independent Variable
# sci - kit - learn (sklearn) library's Column Transformer and OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
column_transformer = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [0])], 
                                        remainder = 'passthrough')
X = np.array( column_transformer.fit_transform(X) )

# Encoding the Dependent Variable
# sci - kit - learn (sklearn) library
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print (y)

# Splitting the dataset into the Training set and Test set
# sci - kit - learn (sklearn) libraries
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, 
                                                     y,
                                                     test_size= 0.2,
                                                     random_state=1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Feature Scaling
# sci-kit-learn (sklearn) libraries
from sklearn.preprocessing import StandardScaler
standard_scalar = StandardScaler()

X_train[:, 3:] = standard_scalar.fit_transform(X_train[:, 3:])
X_test[:, 3:] = standard_scalar.transform(X_test[:, 3:])
print(X_train)
print(X_test)

print("Hello")

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# sci-kit learn model selection train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, 
                                                     y, 
                                                     test_size = 0.25, 
                                                     random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform( X_train )
X_test = standard_scaler.transform( X_test )
print( X_train )
print( X_test )

# Training the Logistic Regression model on the Training set
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

from sklearn.linear_model import LogisticRegression
# random_state == 0 is the generator seed == 0
classifier_logisticRegression = LogisticRegression(random_state = 0)
classifier_logisticRegression.fit( X_train, 
                                   y_train )

# Predicting a new result
# 30 years old and 87000 dollars.
print( classifier_logisticRegression.predict( standard_scaler.transform([[30,87000]]) ))

# Predicting the Test set results
y_predected = classifier_logisticRegression.predict(X_test)
print(np.concatenate(( y_predected.reshape(len(y_predected),1), 
                       y_test.reshape(len(y_test),1)),
                       1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
confustion_matrix = confusion_matrix(y_test, y_predected)
print( confustion_matrix )
accuracy_score( y_test, 
                y_predected )

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = standard_scaler.inverse_transform(X_train), y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))

plt.contourf(X1, X2, classifier_logisticRegression.predict(standard_scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter( X_set[y_set == j, 0], 
                 X_set[y_set == j, 1], 
                 c = ListedColormap(('red', 'green'))(i), 
                 label = j)
    
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = standard_scaler.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier_logisticRegression.predict(standard_scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim( X1.min(), 
          X1.max() )
plt.ylim( X2.min(), 
          X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter( X_set[y_set == j, 0], 
                 X_set[y_set == j, 1], 
                 c = ListedColormap(('red', 'green'))(i), 
                 label = j)

plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
print("Hello")
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

print ('The size of digits is: ',len(digits.data))
'''
print(digits.data)
print(digits.target)
print(digits.images[0])
'''

''' Train the data and leave the last 10 for testing'''
clf = svm.SVC(gamma=0.0001, C=100)
x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

''' Test the last 10'''
iCnt=-4
print('Prediction:',clf.predict(digits.data[iCnt].reshape(1, -1)))
plt.imshow(digits.images[iCnt],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()

print("Got this far")




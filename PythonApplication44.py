#### 3D SVM

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()

X = iris.data[:, :3]
Y = iris.target
X = X[np.logical_or(Y==0 , Y ==1)]#Features of the first two groups
Y = Y[np.logical_or(Y==0 , Y ==1)]#The targets of the first two groups
print('_____________________')
print(Y)
model = svm.SVC(kernel = 'linear')
model.fit(X,Y)
#lambda=The function has a  one line that takes x and y
z = lambda x,y :(-model.intercept_ - model.coef_[0][0]*x - model.coef_[0][1]*y/model.coef_[0][2])

temp = np.linspace(-5,5,30)
x,y = np.meshgrid(temp, temp)
fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')
ax.plot3D(X[Y==0, 0] , X[Y==0,1], X[Y==0,2], 'ob')
ax.plot3D(X[Y==1, 0] , X[Y==1,1], X[Y==1,2], 'sr')

ax.plot_surface(x,y,z(x,y))
plt.show()

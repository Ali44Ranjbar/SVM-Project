#SVM:
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x=[1,5,1.5,8,1  ,9,2   ,8 , 2.3,7,2.4,8]
y=[2,8,1.8,8,0.6,11,1.6,7.7,1.6,7,1.6,7]

plt.scatter(x,y)
plt.show()

lst=[]

for i in range(0 , len(x)):
    lst.append([ x[i] , y[i]])
#[[]]

X=np.array(lst)
Y=["A","B","A","B","A","B","A","B","A","B","A","B"]

model=svm.SVC( kernel="linear") #kernel=هسته
model.fit(X,Y)

print("Predict of [ 1.5 , 1.5] is",model.predict([[1.5,1.5]]))
print("Predict of [ 10 , 10] is",model.predict([[10,10]]))
 
w=model.coef_[0]
intercept=model.intercept_[0]
print(w[0],w[1] , intercept)
xx=np.linspace(0,10 ,200)
print(xx)

yy=(-w[0]/w[1]) * xx - intercept / w[1]
print(yy)

plt.scatter(X[: ,0] , X[: ,1] , c="b")
plt.plot(xx,yy, c="k")
plt.show()


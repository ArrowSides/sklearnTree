#coding=utf8
import numpy as np
import matplotlib.pyplot as plt
 
 
from sklearn import tree
from sklearn.datasets import load_iris

#载入iris数据集
iris = load_iris()

#print(iris.data)

#选用第一个和第三个特征作为X
X = iris.data[:,[0,2]]


#选用target作为label
y = iris.target

#print(y)

#设定最大深度为4 的分类决策树
clf = tree.DecisionTreeClassifier(max_depth=4)
 
#拟合数据
clf = clf.fit(X,y)

#提取特征的min和max
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 
#一维数组np.meshgrid生成网格点坐标矩阵xx和yy
#第一列花萼长度数据按h取等分作为行，并复制多行得到xx网格矩阵
#再把第二列花萼宽度数据按h取等分，作为列，并复制多列得到yy网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
 
#调用ravel()函数将xx和yy的两个矩阵转变成一维数组
#调用np.c_[]函数组合成一个二维数组进行预测
#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

print(clf.apply(np.c_[xx.ravel(), yy.ravel()]))
print(clf.get_n_leaves())

#print(X.shape)

#print(y.size)

#print(xx.shape)

#print(Z.shape)
#调用reshape()函数修改形状，将其Z转换为两个特征（长度和宽度）
#Z = Z.reshape(xx.shape)

#print(Z.shape)
 
#plt.contourf绘制等高线
#plt.contourf(xx, yy, Z, alpha=0.4)
#plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
#plt.show()

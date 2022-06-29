import torch
from sklearn import datasets
from sklearn.datasets import make_classification
import torch

iris = datasets.load_iris() # 导入数据集
X = iris.data # 获得其特征向量
y = iris.target # 获得样本label
X = torch.tensor(X)
y = torch.tensor(y)
print(X)
print(y)

X, y = make_classification(n_samples=10, n_features=5, n_informative=2,
    n_redundant=2, n_classes=2, n_clusters_per_class=2, scale=1.0,
    random_state=20)

for x_,y_ in zip(X,y):
    print(y_,end=': ')
    print(x_)


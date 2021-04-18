import math
import matplotlib as mpl
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

n_features = 2  # 每个样本有几个属性或特征
x, y = make_blobs(n_samples=300, n_features=n_features, centers=6)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)



clf = ExtraTreesClassifier(n_estimators=10, max_features=math.sqrt(n_features), max_depth=None, min_samples_split=2,
                            bootstrap=False)

clf.fit(x_train, y_train)

x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点行列均为200点
area_smaple_point = np.stack((x1.flat, x2.flat), axis=1)  # 将区域划分为一系列测试点去用学习的模型预测，进而根据预测结果画区域

area3_predict = clf.predict(area_smaple_point)
area3_predict = area3_predict.reshape(x1.shape)

mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

classifier_area_color = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 区域颜色
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])  # 样本所属类别颜色

plt.pcolormesh(x1, x2, area3_predict, cmap=classifier_area_color)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='o', s=50, cmap=cm_dark)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, marker='x', s=50, cmap=cm_dark)

plt.xlabel('data_x', fontsize=8)
plt.ylabel('data_y', fontsize=8)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'ExtraTreesClassifier:极端随机树', fontsize=8)
plt.text(x1_max - 9, x2_max - 2, u'$o---train ; x---test$')

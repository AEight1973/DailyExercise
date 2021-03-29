from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib.font_manager import FontProperties
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy


def run(x, y):
    # 1. 确定自变量和因变量
    # x = file['广告费用']
    n = len(x)
    p = 1

    # 2. 画散点图
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    plt.style.use('ggplot')

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.xlabel('价格差/万元', fontproperties=font)
    plt.ylabel('销量/万', fontproperties=font)
    plt.title('价格差和销量关系', fontproperties=font, size=15)
    plt.scatter(x, y)

    # 3. 求解模型参数，建立回归方程
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)  # 给模型传入自变量，因变量。需要将自变量的sr改为ndarray。方法二：model.fit(np.array(x).reshape(-1, 1), y)
    print('coefficient:', model.coef_)  # 斜率 coefficient系数
    print('intercept:', model.intercept_)  # 截距
    ypred = model.predict(x[:, np.newaxis])  # y的预测值

    ax2 = fig.add_subplot(1, 2, 2)
    plt.scatter(x, y)
    plt.plot(x, ypred)
    plt.title('价格差回归方程', fontproperties=font, size=15)
    plt.ylabel('价格差', fontproperties=font)
    plt.xlabel('销量', fontproperties=font)
    # plt.show()

    # 4. 模型评估，好的标准：判定系数接近; mse等较小
    # 4.1 判定系数coefficient of determination
    # 4.1.1 手动计算判定系数
    SSR = ((ypred - y.mean()) ** 2).sum()
    print('ssr:', SSR)

    SST = ((y - y.mean()) ** 2).sum()
    print('sst:', SST)

    SSE = ((y - ypred) ** 2).sum()
    print('sse:', SSE)

    r2 = SSR / SST
    print('判定系数=ssr/sst:', r2)

    # 4.1.2 model.score()自动计算判定系数
    # r2 = model.score(x[:, np.newaxis], y)
    # print(metrics.r2_score(y, ypred))

    # 4.2 均方误差mse
    MSE = SSE / (n - p - 1)  # sse的自由度不应该是n-p-1么，但是只有当分母是n才和mean_squared_error计算结果一致
    print('MSE:', MSE)
    # print('MSE:', metrics.mean_squared_error(y, ypred))

    # 4.3 均方根误差RMSE
    print('均方根误差RMSE:', MSE ** 0.5)

    # 4.4 平均绝对误差
    MAE = metrics.mean_absolute_error(y, ypred)
    print('MAE:', MAE)

    # 5. 显著性检验
    # 5.1 线性相关变量的显著性检验，皮尔逊相关系数
    # print(scipy.stats.pearsonr(x, y))  # 返回r值和相关系数假设检验的p值。r越大表示线性相关程度越高。p值越小，表示原假设“总体相关系数=0的可能性越小，r值可信度越高。

    # 5.2 线性or非线性相关变量的显著性检验
    # 5.2.1 F检验
    MSR = SSR / p
    F = MSR / MSE
    F_p = scipy.stats.f.sf(F, p, n - 2)
    print('F检验的p值', F_p)

    # 5.2.2 t检验
    s_coef = np.sqrt(MSE) / np.sqrt(n * np.var(x))
    t = model.coef_ / s_coef
    print(t)
    t_p = scipy.stats.t.sf(t, n - 2)
    print('t检验的p值：', t_p)

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score  # 划分数据集的方法
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import r2_score
import time

start = time.time()
#导入数据函数
def load_housing_data():
    return pd.read_csv('../../data/data_Lu/final_train.csv')


def load_housing_data1():
    return pd.read_csv('../../data/data_Lu/final_test.csv')


#划分数据集函数
def split_train_test(data,test_ratio):
    indices = np.random.permutation(len(data)) #随机全排列
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

housing = load_housing_data()
housing1 = load_housing_data1()
#显示数据集
#print(housing.head)
#显示数据信息（数据量，数据类型，内存占用）
#print(housing.info())
#查看数据分布详细信息
#print(housing.describe())
#绘制图
#housing.hist(bins=50,figsize=(15,10))#bins 柱子个数
#plt.savefig('../../reports/figures/data.png')
#plt.show()
#划分数据集
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
#print("train set len",len(train_set))   #1165
#print("test set len",len(test_set))     #292
#print(test_set['SalePrice'])
#print(train_set['SalePrice'])
#划分目标与训练
target = 'SalePrice'
ID = 'Id'

x_columns = [x for x in train_set.columns if x not in [target, ID]]
housing_prepared = train_set[x_columns]
#print(len(housing_prepared))
housing_price = train_set['SalePrice']
#训练模型
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_price)
housing_predictions = forest_reg.predict(housing_prepared)      #利用模型预测房价
#forest_mse = mean_squared_error(housing_price, housing_predictions)     #计算误差均方差
#forest_rmse = np.sqrt(forest_mse)                                       #计算误差标准差
#print("forest_rmse误差标准差\n", forest_rmse)
#forest_scores = cross_val_score(forest_reg, housing_prepared, housing_price,    #进行交叉验证
#                                scoring="neg_mean_squared_error", cv=10)        #获得每次验证所得方差
##forest_scores = cross_val_score(forest_reg, housing_prepared, housing_price,    #进行交叉验证
                                ##scoring="r2", cv=10)                            #获得R平方值，显示方程对观测值的拟合程度如何
##forest_scores_mean = np.mean(forest_scores)                                     #求R平方值的均值
##print("forest_scores_mean交叉验证后得分平均值\n", forest_scores_mean)
#print("forest_rmse_scores.mean\n", forest_rmse_scores.mean())
#print("forest_rmse_scores.std\n", forest_rmse_scores.std())
#对模型进行优化，调参
param_grid = [
    # 12 (3×4) 种超参数组合
    # n_estimators:决策树个数,越多越好，
    # max_features:选择最适属性时划分的特征不能超过此值
    # bootstrap;是否有放回的抽样
    {'n_estimators': [500, 1000], 'max_features': [500, 1000]}, #组合1
    #  6 (2×3) 种
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [50, 65,75, 85]},#组合2
  ]
print("开始计算每个参数对应分数")
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='r2')#scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_price)
print("找到了最佳参数")
print("The best parameters are最佳参数组合\n", grid_search.best_params_)
print(grid_search.best_estimator_)                          #查看最好分类器
print(grid_search.best_score_)                              #查看最高得分
final_model = grid_search.best_estimator_
#在测试集上进行测试
X_test = test_set.drop(["SalePrice"], axis=1)          #测试集受训数据
y_test = test_set["SalePrice"].copy()                       #测试集评价数据
##这个时候只需要transform
##X_test= test_set.transform(X_test)                        #将数据实现标准化
final_predictions = final_model.predict(X_test)
#print("在训练集测试部分的预测结果", final_predictions)
##均方误差
#final_mse = mean_squared_error(y_test, final_predictions)
#print("final_mse均方误差\n", final_mse)               #离散程度
#final_rmse = np.sqrt(final_mse)
#print("final_rmse标准差\n", final_rmse)              #离散程度
final_scores = cross_val_score(final_model, X_test, y_test, scoring='r2', cv=10)#交叉验证
#forest_rmse_scores = np.sqrt(-forest_scores)
print("最佳参数组合下交叉验证得分数组", final_scores)
final_scores_mean = np.mean(final_scores)
print("交叉验证得分数组均值", final_scores_mean)
print("最终得分", r2_score(y_test, final_predictions))
#print(len(y_test))
#print(len(final_predictions))
#np.savetxt("train_prediction.csv", final_predictions)
#用测试数据集来测试
test_predictions = final_model.predict(housing1)
final_test_predictions = np.exp(test_predictions)
#print(len(housing1))
#print(len(final_test_predictions))
np.savetxt("test_predictionR3.csv", final_test_predictions)
end = time.time()
print("run time is", end-start)
#joblib.dump(final_model,'best_model.pkl')




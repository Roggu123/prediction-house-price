import pandas as pd
import numpy as np
from scipy.stats import skew

df_test = pd.read_csv("../../test.csv")
df_train = pd.read_csv("../../train.csv")

TARGET = 'SalePrice'

#数据处理

y_train = np.log(df_train[TARGET]+1)
#print(df_train[TARGET])
#print(y_train)
df_train.drop([TARGET], axis=1, inplace=True)

#合并数据
all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],df_test.loc[:,'MSSubClass':'SaleCondition']))
#print(all_data)df_

#用对数转换将倾偏态特征转换成正态
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index #转换类型
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) #计算倾斜度
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index # 选择倾斜度大于0.75的特征
#由于选中的数据倾斜度较高，呈现偏态分布
#对选中的数据进行对数变换
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
#离散特征 one-hot 编码
all_data = pd.get_dummies(all_data)

#处理缺失值
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#对于缺失值，不同情况不同分析  高特征的低缺失值可以尝试填充估计；高缺失值的可以通过回归估计计算
#低特征的低缺失值可以不做处理；高缺失值的可直接剔除字段
#通过观察发现出现缺失值的字段的相关系数都很低，特征都不明显，因此可以删除
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

train = all_data[:df_train.shape[0]]
test = all_data[:df_test.shape[0]]




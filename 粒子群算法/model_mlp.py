import os
import pandas as pd
from pandas import  DataFrame
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


'''
:param path csv文件路径
:param columns 选取指定的列名
'''
def concatCsv(path, columns, dropColumns):
    files = os.listdir(path)  # 获取文件夹下所有文件名
    df1 = pd.read_csv(path + '/' + files[0], encoding='gbk')
    if dropColumns:
        #ether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
        df1 = df1.drop(dropColumns, axis=1)
        for file in files[1:]:
            df2 = pd.read_csv(path + '/' + file, encoding='gbk')# 打开csv文件，注意编码问题，保存到df2中
            df2 = df2.drop(dropColumns, axis=1)
            df1 = pd.concat([df1, df2], sort=False)  # 将df2数据与df1合并
    else:
        df1 = pd.read_csv(path + '/' + files[0], encoding='gbk')[columns]
        for file in files[1:]:
            df2 = pd.read_csv(path + '/' + file, encoding='gbk')[columns]  # 打开csv文件，注意编码问题，保存到df2中
            df1 = pd.concat([df1, df2], axis=0, ignore_index=True)[columns]  # 将df2数据与df1合并
    # df1.to_csv(path + '/' + 'total.csv')  # 将结果保存为新的csv文件
    return df1

# 统计数据 的总体情况 包括 最大值  最小值  均值 和中位数 CV变异系数=标准差/均值
def dataSummary(df):
    return pd.DataFrame({"max": data.max(), "min": data.min(), "mean": data.mean(), "median": data.median(), "CV": data.std()/data.mean()})
# 数据直方图 输入dataFrame 输出画板对象
def dataPlt(df,method):
    fig = plt.figure(figsize=(400, 300))
    columnLength =len(df.columns)
    h = (columnLength/2 if (columnLength % 2 == 0) else columnLength/2+1)
    for index in range(columnLength):
        col = df.columns[index]
        ax1 = fig.add_subplot(2, h, index+1)
        if method == "hist":
            ax1.hist(np.array(data[col].values), bins=np.linspace(data[col].min(), data[col].max(), 10))
        elif method == "boxPlot":
            ax1.boxplot(np.array(data[col].values))
        ax1.set_title(col)
    return plt


#盖帽法实现 去除异常值 将3倍标准差的数据过滤调
def removeOutlier(df):
    for col in df.columns:
        df = df.loc[abs(df[col]-df[col].mean()) < 3 * df[col].std()]
    return df

# 计算rmse
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)
    n = len(y_pred)
    return np.sqrt(sum_sq/n)

# 计算mse
def mse(y_ture, y_pred):
    return mean_squared_error(y_ture, y_pred)

# 计算平均绝对误差
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

path = 'D:/python project/well_optimizer/data/data12.25'  # 设置csv所在文件夹
#columns = ['ROPmhr', 'WOBklbs', 'FLWpmpslmn', 'RPM', 'TORQUElbft', 'SPPpsi','']
dropColumns=['TotDepth','ROPA']
data = concatCsv(path,[],dropColumns) # 合成path目录下的所有csv文件

#处理异常值,如果一行的数据中，存在某一列有缺失值，就删除改行
data.dropna(how="any")
# 统计基本数据  最大值 最小值 均值  中位数 data.describe()
# print(dataSummary(data))

# 盖帽法处理
filterData = removeOutlier(data)

# 去除目标列作为自变量
data_x = filterData.drop(['ROPmhr', 'TORQUElbft', 'SPPpsi'],axis=1)

# 自变量数据标准化
std = StandardScaler()
x_std = std.fit_transform(data_x)
x_std = DataFrame(x_std, index=data_x.index, columns=data_x.columns.values)

# 标准化后数据还原
origin_data = MinMaxScaler()


# 用变量除以他们的标准误就可得到学生化数值
# data_x = (data_x-data_x.mean())/(data_x.std())

# 目标变量
data_rop_y = filterData['ROPmhr']
data_tor_y = filterData['TORQUElbft']
data_spp_y = filterData['SPPpsi']

# mlp模型
mlpr = MLPRegressor(hidden_layer_sizes=(100, ), activation='tanh', solver='adam'
                    , alpha=0.0001, batch_size='auto', learning_rate='constant'
                    , learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True
                    , random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9
                    , nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9
                    , beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)


#rop
# 划分训练集和验证集
X_rop_train, X_rop_valid, y_rop_train, y_rop_valid = train_test_split(x_std, data_rop_y, test_size=0.30, random_state=100)

mlpr.fit(X_rop_train,y_rop_train)
y_pred_rop = mlpr.predict(X_rop_valid)
rmse_rop = rmse(y_rop_valid,y_pred_rop)
mse_rop = mse(y_rop_valid,y_pred_rop)
mae_rop = mae(y_rop_valid,y_pred_rop)

print(rmse_rop)
print(mae_rop)

# # # tor
# X_tor_train, X_tor_valid, y_tor_train, y_tor_valid = train_test_split(data_x, data_tor_y, test_size=0.3, random_state=100)
#
# mlpr.fit(X_tor_train,y_tor_train)
# pred_tor_y = mlpr.predict(X_tor_valid)
# rmse_tor = rmse(y_tor_valid,pred_tor_y)
# mse_tor = mse(y_tor_valid,pred_tor_y)
# print(rmse_tor)
# #
# # #spp
# X_spp_train, X_spp_valid, y_spp_train, y_spp_valid = train_test_split(data_x, data_spp_y, test_size=0.3, random_state=100)
#
# mlpr.fit(X_spp_train,y_spp_train)
# pred_spp_y = mlpr.predict(X_spp_valid)
# rmse_spp = rmse(y_spp_valid,pred_spp_y)
# mse_spp = mse(y_spp_valid,pred_spp_y)
# print(rmse_spp)






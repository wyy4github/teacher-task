import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

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

#盖帽法实现 去除异常值 将3倍标准差的数据过滤调
def removeOutlier(df):
    for col in df.columns:
        df = df.loc[abs(df[col]-df[col].mean()) < 3 * df[col].std()]
    return df

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

# 盖帽法处理
filterData = removeOutlier(data)

# 去除目标列作为自变量
data_x = filterData.drop(['ROPmhr', 'TORQUElbft', 'SPPpsi'],axis=1)

# 自变量数据标准化
std = StandardScaler()
x_std = std.fit_transform(data_x)
x_std = DataFrame(x_std, index=data_x.index, columns=data_x.columns.values)

# 目标变量
data_rop_y = filterData['ROPmhr']
data_tor_y = filterData['TORQUElbft']
data_spp_y = filterData['SPPpsi']

# 随机抽取数据作为测试集
randomData = filterData.sample(frac=0.20, replace=False, random_state=1)
test_rop_y = randomData['ROPmhr']
randomData = randomData.drop(['ROPmhr', 'TORQUElbft', 'SPPpsi'],axis=1)
randomData_x = std.fit_transform(randomData)
test_rop_x = DataFrame(randomData_x, index=randomData.index, columns=randomData.columns.values)


X_rop_train, X_rop_valid, y_rop_train, y_rop_valid = train_test_split(x_std, data_rop_y, test_size=0.30, random_state=100)
y_rop_train = pd.DataFrame(y_rop_train)
y_rop_valid = pd.DataFrame(y_rop_valid)


# 定义数据集相关的常数
input_node = 23
output_node = 1

# 配置神经网络的参数
layer_node = 20  # 隐藏层节点数
learning_rate_base = 0.01  # 基础学习率
learning_rate_decay = 0.09  # 基础学习率的衰减率
regularization_rate = 0.0001  # 描述模型复杂度
training_step = 9500  # 训练轮数
moving_average_decay = 0.99  # 滑动平均衰减率

# 辅助函数，计算神经网络的前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        # 当没有滑动平均类时，直接使用参数当前取值
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1) + avg_class.average(biases1)))
        return tf.matmul(layer1, avg_class.average(weights2) + avg_class.average(biases2))


# 训练模型的过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None, input_node), name='x-input')
y = tf.compat.v1.placeholder(tf.float32, shape=(None, output_node), name='x-input')
# 生成隐藏层的参数
weights1 = tf.Variable(tf.random.normal([input_node, layer_node], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[layer_node]))
# 生成输出层的参数
weights2 = tf.Variable(tf.random.normal([layer_node, output_node], stddev=0.11))
biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
# 计算当前参数下神经网络前向传播的结果
# 不使用滑动平均值
y_ = inference(x, None, weights1, biases1, weights2, biases2)

global_step = tf.Variable(0)

# 自定义损失函数
cross_entropy_mean = tf.reduce_mean(tf.square(y-y_))

# 计算L2正则化损失函数
regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
regularization = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularization

# 设置指数衰减学习率
learning_rate = tf.compat.v1.train.exponential_decay(learning_rate_base, global_step, training_step, learning_rate_decay)
# 优化器的使用
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.compat.v1.Session() as sess:
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    train_feed = {x: X_rop_train, y: y_rop_train}
    validation_feed = {x: X_rop_valid, y: y_rop_valid}
    test_feed = {x: test_rop_x}
    for i in range(training_step):
        if i % 100 == 0:
            valloss = sess.run(loss, feed_dict=validation_feed)
            print("for i:{} step loss is {}".format(i, valloss))
        sess.run(train_step, feed_dict=train_feed)
    y_pred = sess.run(y_, feed_dict=test_feed)
    print(mae(test_rop_y,y_pred))
    print(mse(test_rop_y,y_pred))
    print(y_pred)


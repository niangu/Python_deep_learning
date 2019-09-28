#TensorFlow的运行方式如下4步：
#（1）加载数据及定义超参数
#（2）构建网络
#（3）训练模型
#（4）评估模型和进行预测

#生成及加载数据：首先生成输入数据，我们假设最后要学习的方程为y=x^2-0。5，我们来构造满足这个方程的一堆x和y,同时加入一些不满足方程的噪声点
import tensorflow as tf
import numpy as np
#构造满足一元二次方程的函数
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] #为了使点更密一些，我们构建了300个点，分布在-1到1区间，直接采用np生成等差数列的方法，并将结果为300个点的一维数组，转换为300x1的二维数组
noise = np.random.normal(0, 0.05, x_data.shape)#加入一些噪音点，使它与x_data的维度一致，并且拟合均值为0，方差为0.05的正态分布
y_data = np.square(x_data) - 0.5 + noise #y = x^2 - 0.5 + 噪声
#接下来定义x和y的占位符来作为将要输入神经网络的变量
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#构建网络模型
#构建一个隐藏层和一个输出层
#作为神经网络中的层，输入参数应该有4个变量：输入数据，输入数据的维度，输出数据的维度和激活函数。
#每一层经过向量化（y = weights*x+biases）的处理，并且并且经过激活函数的非线性化处理后，最后得到输出数据

#定义隐藏层和输出层：
def add_layer(inputs, in_size, out_size, activation_function=None):
    #构建权重：in_size*out_size大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #构建偏置：1*out_size的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]))
    #矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs #得到输出数据

#构建隐藏层，假设隐藏层有10个神经元
h1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu)
#构建隐藏层，假设输出层和输入层一样，有1个神经元
prediction = add_layer(h1, 20, 1, activation_function=None)

#接下来需要构建损失函数：计算输出层的预测值和真实值间的误差，对二者的平方求和在取平均，得到损失函数。运用梯度下降法，以0.1的效率最小化损失：
#计算预测值和真实值间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#训练模型
#我们让TensorFlow训练1000次，每50次输出训练的损失值
init = tf.global_variables_initializer() #初始化🗂所有变量
sess = tf.Session()

sess.run(init)

for i in range(1000):#训练1000次
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:#每50次打印一次损失值
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


#所谓超参数，就是指机器学习模型里的框架参数，与权重参数不同的是，它是需要手动设定，不断试错的
#学习率：最常设定的超参数，学习率设置得越大，训练时间越短，速度越快， 学习率越小，训练准确度越高
#先设置0.01，观察损失值的变化，然后尝试0.001,0.0001,最终确定一个比较合适的学习率
#mini-batch大小是另一个最常设定的超参数，每批大小决定了权重的更新规则，例如：
#大小为32时，就是把32个样本的梯度全部计算完，然后求平均值，去更新权重。批次越大训练的速度越快，可以利用矩阵，线性代数库来加速，但是权重更新频率略低。批次越小训练的速度就慢
#结合机器的硬件性能以及数据集大小来设定

#正则项系数：常用超参数：一般凭经验。一般来说，如果在比较复杂的网络发现出现了明显的过拟合（训练数据准确率很高，测试数据准确率反而下降），可以考虑增加此项
#初学可以设置为0，确定好一个比较好的学习率后，在给一个值，根据准确率进行精确调整




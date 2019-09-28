#加载数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)
#构建模型
#设置训练的超参数：学习率，训练的轮数（全部数据训练完一遍），每次训练的数据多少，每隔多少轮显示一次结果
learning_rate = 0.01#学习率
training_epochs = 20#训练的轮数
batch_size = 256#每次训练的数据多少
display_step = 1#每隔多少轮显示一次训练的结果

#设置其他参数变量
examples_to_show = 10#表示从测试集中选择10张照片取验证自动编码器的结果

#初始化权重与定义网络结构：俩个隐藏层，第一个隐藏层神经元为256个，第二层隐藏神经元为128个
#网络参数
n_hidden_1 = 256#第一个隐藏神经元个数，也是特征值个数
n_hidden_2 = 128#第二个隐藏层神经元个数，也是特征值个数
n_input = 784#输入数据的特征值个数：28 * 28= 784

#然后定义输入数据：无监督学习，只需要输入图片数据，不需要标记数据
X = tf.placeholder("float", [None, n_input])

#初始化每一层的权重和偏置
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
#自动编码模型的网络结构：压缩，解压
#定义压缩函数
def encoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

#定义解压函数
def decoder(x):
    #Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    #Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

#构建模型
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

#接着我们构建损失函数和优化器，这里损失函数用"最小二乘法"对原始数据集和输出的数据集进行平方差并取均值运算：优化器采用RMSPropOptimizer
#得出预测值
y_pred = decoder_op
#得出真实值，即使输入值
y_true = X
#定义损失函数和优化器
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#初始化变量
init = tf.global_variables_initializer()
#训练数据及评估模型
#在一个会话中启动图，开始训练和评估
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    #开始训练
    for epoch in range(training_epochs):
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            #每一轮，打印一次损失值
            if epoch % display_step == 0:
                print("Epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    #对测试集应用训练好的自动编码网络
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    #比较测试集原始图片和自动编码网络的重建结果
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))

    f.show()
    plt.show()
    plt.waitforbuttonpress()
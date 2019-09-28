#加载数据
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
#接着需要处理输入的数据，把上述trX,teX的形状变为[-1,28,28,1],-1表示不考虑输入图片的数量，28x28是图片长和宽的像素数1是通道（channel）数量，因为MNIST的图片是黑白的，所以通道是一，如果是RGB彩色图像，通道是3
trX = trX.reshape(-1, 28, 28, 1) #28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)#28x28x1 input img
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])
#初始化权重与定义网络结构。这里我们将要构建一个拥有3个卷积层和3个池化层，随后接一个全连接层和1个输出层的卷积神经网络
#首先定义初始化权重的函数
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

#初始化权重方法如下，我们设置卷积核的大小为3x3:
w = init_weights([3, 3, 1, 32]) #patch大小为3x3，输入维度为1,输出维度为32
w2 = init_weights([3, 3, 32, 64])#patch大小为3x3，输入维度为32，输出维度为64
w3 = init_weights([3, 3, 64, 128])#patch大小为3✖3,输入维度为64，输出维度为128
w4 = init_weights([128 * 4 * 4, 625])#全连接层，输入维度为128x4x4，是上一层的输出数据由三维转变为一维，输出维度为625
w_o = init_weights([625, 10])#输出层，输入维度为625，输出维度为10，代表10类（labels）

#定义一个模型函数
#神经网络模型的构建函数，传入以下参数
#X：输入数据
#w：每一层的权重
#p_keep_conv, p_keep_hidden:dropout要保留的神经元比例
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    #第一组卷积层及池化层，最后dropout一些神经元
    lla = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    #lla shape=(?,28,28,32)
    l1 = tf.nn.max_pool(lla, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #ll shape=(?, 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    #第二组卷积层及池化层，最后dropout一些神经元
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    #l2a shape(?, 14, 14, 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #l2 shape=(?, 7, 7, 4)
    l2 = tf.nn.dropout(l2, p_keep_conv)

    #第三组卷积层及池化层，最后dropout一些神经元
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    #l3a shape=(?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #l3 shape=(?, 4, 4, 128)
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]]) #reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    #全连接层，最后dropout一些神经元
    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    #输出层
    pyx = tf.matmul(l4, w_o)
    return pyx #返回预测值

#我们定义dropout的占位符号----keep_conv，它表示在一层中有多少比例的神经元被保留下来，生成网络模型，得到预测值
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)#得到预测值
#接下来，定义损失函数，这里我们仍然采用tf.nn.softmax_cross_entropy_with_logits来比较预测值和真实值的差异，并且做均值处理；定义训练的操作（train_op）,
#采用实现RMSProp算法的优化器tf.train.RMSPropOptimizer,学习率为0.001，衰减值为0.9,使损失最小，定义预测的操作(predict_op):

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

#训练和评估模型
#先定义训练时的批次大小和评估时的批次大小
batch_size = 128
test_size = 256
#在一个会话中启动图，开始训练和评估:
#Lauch the graph in a session
with tf.Session() as sess:
    #you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_indices = np.arange(len(teX)) #Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))


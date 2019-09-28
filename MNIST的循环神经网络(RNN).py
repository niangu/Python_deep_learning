#加载数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#加载数据
mnist = input_data.read_data_sets('data/', one_hot=True)

#构建模型
#设置训练的超参数
lr = 0.001
training_iters = 100000
batch_size = 128

#为了使用RNN来分类图片，把每张图片的行看成是一个像素序列（sequence）.因为MNIST的图片大小是28X28像素，所以我们把每一个图像样本看成一行行的序列。
#因此共有（28个元素的序列）*(28行)，然后每一步输入的序列长度是28，输入的步数是28步。
#定义RNN的参数
#神经网络的参数
n_input = 28 #输入层的n
n_steps = 28 #28长度
n_hidden_units = 128 #隐藏层的神经元个数
n_classes = 10 #输出的数量，即分类的类别，0-9个数字，共有10个

#定义输入数据及权重：
#输入数据占位符
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

#定义权重
weights = {
    #(28, 128)
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    #(128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    #(128, )
    'in': tf.Variable(tf.random_normal([n_hidden_units, ])),
    #(10, )
    'out': tf.Variable(tf.random_normal([n_classes, ]))
}
#定义RNN模型
def RNN(X, weights, biases):
    #把输入的X转换成X==>（128 batch * 28 steps inputs）
    X = tf.reshape(X, [-1, n_input])

    #进入隐藏层
    #X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    #X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    #这里采用基本的LSTM循环网络单元：basic LSTM Cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0,
                                             state_is_tuple=True)
    #初始化为0值，lstm单元由俩个部分组成：（c_state, h_state）
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    #dynamic_rnn接收张量（batch, steps, inputs）或者（steps, batch, inputs）作为X_in
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in,
                                             initial_state=init_state,
                                             time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    return results

#定义损失函数和优化器，优化器采用AdamOptimizer
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                              labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
#定义模型预测结果及准确率计算方法：
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#训练数据评估模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
            }))
            step += 1

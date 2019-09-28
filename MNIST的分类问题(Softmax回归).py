#加载数据
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#加载数据
mnist = input_data.read_data_sets('/home/niangu/桌面/TensorFlow/data', one_hot=True)
#使用one_hot的直接原因是，我们使用0-9个类别的多分类的输出层是softmax层，它的输出是一个概率分布，从而要求输入的标记也以概率分布的形式出现，进而可以计算交叉熵
#构建回归模型：我们需要输入原始真实值（group truth）,计算采用softmax函数拟合后的预测值，并且定义损失函数和优化器
#定义回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b #预测值
#在这里，我们要求TensorFlow用梯度下降算法以0.5的学习率最小化交叉熵，这里也可以才采用其他优化器，只需要调整tf.train.GradientDescentOptimizer即可
#定义损失函数和优化器
y_ = tf.placeholder(tf.float32, [None, 10])#输入真实值的占位符

#我们用tf.nn.softmax_cross_entropy_with_logits来计算预测值y与真实值y_的差值，并取均值
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#采用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#训练模型
#在训练之前初始化我们创建的变量，以及在会话中启动模型
#这里使用InteractiveSession()来创建交互式上下文的TensorFlow绘画
#与常规会话不同的是，交互式会话成为默认会话
#方法（如tf.Tensor.eval和tf.Operation.run）都可以使用该会话来运行操作（OP）
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#我们让模型循环训练1000次，在每次循环中我们都随机抓取训练数据中100个数据点，来替换之前的占位符号
#Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    #这种训练方式称为随机训练，使用SGD方法进行随机梯度下降，也就是每次从训练数据集中随机抓取一小部分数据进行梯度下降训练。
#评估模型
#tf.argmax(y, 1)返回的是模型对任一输入x预测到的标记值，tf.argmax(y_,1)代表正确的标记值。我们用tf.equal来检测预测值和真实值是否匹配，并且将预测后得到的布尔值转化成浮点数，并且取平均值
#评估训练好的模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))#计算预测值和真实值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#布尔型转化为浮点数，并且取平均值，得到准确率
#计算模型在测试集上的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))

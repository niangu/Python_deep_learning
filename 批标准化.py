import tensorflow as tf
'''
#计算Wx_plus_b的均值和方差，其中axes=[0]表示想要标准化的维度
fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0],)
scale = tf.Variable(tf.ones([out_size]))
shift = tf.Variable(tf.zeros([out_size]))
epsilon = 0.001
Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)
#也就是在做：
#Wx_plus_b = (Wx_plus_b-fc_mean)/tf.sqrt(fc_var + 0.001)
#Wx_plus_b = Wx_plus_b * scale + shift
'''
#sigmoid函数
a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
sess = tf.Session()
print(sess.run(tf.sigmoid(a)))

#relu
a = tf.constant([-1.0, 2.0])
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))

a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
with tf.Session() as sess:
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 4])
    print(sess.run(b))
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 1])
    print(sess.run(b))



import tensorflow as tf
'''
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")
'''
#张量
a = tf.constant([5, 3], name="input_a")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(c, b, name="add_d")
sess = tf.Session()
#print(sess.run(e))
writer = tf.summary.FileWriter('/home/niangu/my_graph', sess.graph)

writer.close()
sess.close()



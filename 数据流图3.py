#名称作用域的基本用法是将Op添加到with tf.name_scope(<name>)语句块中
import tensorflow as tf
'''
with tf.name_scope("Scope_A"):
    a = tf.add(1, 2, name="A_add")
    b = tf.multiplytiply(a, 3, name="A_multiply")

with tf.name_scope("Scope_B"):
    c = tf.add(4, 5, name="B_add")
    d = tf.multiplytiply(c, 6, name="B_multiply")

e = tf.add(b, d, name="output")

writer = tf.summary.FileWriter('/home/niangu/my_graph', graph=tf.get_default_graph())
writer.close()
'''

graph = tf.Graph()
with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[], name="input_a")
    in_2 = tf.placeholder(tf.float32, shape=[], name="input_b")
    const = tf.constant(3, dtype=tf.float32, name="static_value")
    with tf.name_scope("Transformation"):
        with tf.name_scope("Transformation"):
            with tf.name_scope("A"):
                #接收in_1,输出一些值
                A_multiply = tf.multiply(in_1, const)
                A_out = tf.subtract(A_multiply, in_1)

            with tf.name_scope("B"):
                #接收in_2,输出一些值
                B_multiply = tf.multiply(in_2, const)
                B_out = tf.subtract(B_multiply, in_2)

            with tf.name_scope("C"):
                #接收A和B的输出, 并输出一些值
                C_div = tf.div(A_out, B_out)
                C_out = tf.add(C_div, const)

            with tf.name_scope("D"):
                #接收A和B的输出,并输出一些值
                D_div = tf.div(B_out, A_out)
                D_out = tf.add(D_div, const)
        #获取C和D的输出
        out = tf.maximum(C_out, D_out)

        writer = tf.summary.FileWriter('/home/niangu/my_graph', graph=graph)
        writer.close()
import tensorflow as tf
'''
v = tf.get_variable(name, shape, dtype, initializer)#通过所给的名字创建或是返回一个变量
tf.variable_scope(<scope_name>)#为变量指定命名空间
'''
#当tf.get_variable_scope().reuse==False时， variable_scope作用域只能用来创建新变量
#tf.get_variable_scope().reuse = True
'''
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v2 = tf.get_variable("v", [1])
assert v.name == "foo/v:0"
'''
#####################################################
with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    #也可以写成：
    #scope.reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1== v

import tensorflow as tf

#创建一个新的数据流图
g = tf.Graph()
with g.as_default():
    #像往常一样创建一些Op;它们将被添加到Graph对象g中
    a = tf.multiply(2, 3)

#获得默认数据流图的句柄
default_graph = tf.get_default_graph()

#正确的实践-------创建新的数据流图,将默认数据流图忽略
g1 = tf.Graph()
g2 = tf.Graph()
g3 = tf.get_default_graph()

with g1.as_default():
    #定义g1中的Op,张量等
    print()
with g2.as_default():
    #定义g2中的Op,张量等
    print()


#######################################
a = tf.add(2, 5)
b = tf.multiply(a, 3)
sess = tf.Session()
#sess = tf.Session(graph=tf.get_default_graph())
sess.run(b)
sess.run([a, b])
#执行初始化Variable对象所需要的计算,但返回值为None
sess.run(tf.initialize_all_variables())


##############################################################
a = tf.add(2, 5)
b = tf.multiply(a, 3)
sess = tf.Session()
#定义一个字典，比如将a的值替换为15
replace_dict = {a:15}
#运行Session对象，将replace_dict赋给feed_dict
sess.run(b, feed_dict=replace_dict)#返回45
sess.close()
#也可以将Session对象作为上下文管理器加以使用,这样当代码离开其作用域后,该Session对象将自动关闭
with tf.Session() as sess:
    print()

###########################################################
a = tf.constant(5)
sess = tf.Session()
#在with语句块中将Session对象作为默认Session对象
with sess.as_default():
    a.eval()
#必须手工关闭Session对象
sess.close()


#tf.placeholder创建占位符
import numpy as np
#创建一个长度为2,数据类型为int32的占位向量
a = tf.placeholder(tf.int32, shape=[2], name="my_input")

#将该占位向量视为其他任意Tensor对象,加以使用
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")

#完成数据流图的定义
d = tf.add(b, c, name="add_d")

#######################################################Variable
my_var = tf.Variable(3, name="my_variable")
add = tf.add(5, my_var)
mul = tf.multiply(8, my_var)
#2x2零矩阵
zeros = tf.zeros([2, 2])
#长度为6的全1向量
ones = tf.ones([6])
#3x3x3的张量,其元素服从0-10的均匀分布
uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
#3x3x3的张量,其元素服从0均值,标准差为2的正态分布
normal = tf.random_normal([3, 3, 3], mean=0.0, stddev=2.0)
#tf.truncated_normal()不会创建任何偏离均值超过2倍标准差的值，防止张量中出现显著不同的元素
#该Tensor对象不会返回任何小于3.0或大于7.0的值
trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)
#默认均值为0, 默认标准差为1.0
random_var = tf.Variable(tf.truncated_normal([2, 2]))

#必须在一个Session对象内对Variable对象进行初始化
init = tf.initialize_all_variables()#Variable对象初始化
sess = tf.Session()
sess.run(init)

#如果只需要对数据流图中定义的一个Variable对象子集初始化,可使用tf.initialize_variables(), 该对象可接收一个要进行初始化的Variable对象列表
var1 = tf.Variable(0, name="initialize_me")
var2 = tf.Variable(1, name="no_initialization")
init = tf.initialize_variables([var1], name="init_var1")
sess = tf.Session()
sess.run(init)

#修改Variable对象的值：Variable.assign()
#创建一个初值为1的Variable对象
my_var = tf.Variable(1)
#创建一个Op,使其在每次运行时都将该Variable对象乘以2
my_var_times_two = my_var.assign(my_var * 2)

#初始化Op
init = tf.initialize_all_variables()
#启动一个会话
sess = tf.Session()
#初始化Variable对象
sess.run(init)
#将Variable对象乘以2,并将其返回
sess.run(my_var_times_two)
#输出：2
#再次相乘
sess.run(my_var_times_two)
##输出L4
#再次相乘
sess.run(my_var_times_two)
##输出：8


#对于Variable对象的简单自增和自减,TensorFlow提供了Variable.assign_add()方法和Variable.assign_sub()方法
#自增1
sess.run(my_var.assign_add(1))
#自减1
sess.run(my_var.assign_sub(1))

#######################################################################
#创建一些Op
my_var = tf.Variable(0)
init = tf.initialize_all_variables()

#启动多个Session对象
sess1 = tf.Session()
sess2 = tf.Session()
#在sess1内对Variable对象进行初始化,以及在同一Session对象中对my_var的值自增
sess1.run(init)
sess1.run(my_var.assign_add(5))
##输出5
#在sess2内做相同的运算，但使用不同的自自增值
sess2.run(init)
sess2.run(my_var.assign_add(2))
##输出：2
#能够在不同Session对象中独立地对Variable对象的值实施自增运算
sess1.run(my_var.assign_add(5))
##输出：10
sess2.run(my_var.assign_add(2))
##输出：4
#如果希望将所有Variable对象的值重置为初始值，则只需要再次调用tf.initialize_all_variables()
#如果只希望对部分Variable对象重新初始化,可调用tf.initialize_variables()
#创建Op
my_var = tf.Variable(0)
init = tf.initialize_all_variables()
#启动Session对象
sess = tf.Session()
#初始化Variable对像
sess.run(init)
#修改Variable对象的值
sess.run(my_var.assign(10))
#将Variable对象的值重置为初始值0
sess.run(init)

#设置Variable对象只可手工修改，不允许使用Optimizer类时，可在创建这些Variable时将其trairable参数设为False
not_trainable = tf.Variable(0, trainable=False)




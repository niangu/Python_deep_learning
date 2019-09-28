import tensorflow as tf
#TensorFlow中主要有俩种队列，既FIFOQueue和RandomShuffleQueue
"""
#创建一个先入先出队列，初始化队列插入0.1,0.2,0.3三个数字
q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1, 0.2, 0.3],))
#定义出队，+1，入队操作
x = q.dequeue()#取出一个值
y = x + 1
q_inc = q.enqueue([y])

#开启一个会话，执行俩次q_inc操作，随后查看队列内容
with tf.Session() as sess:
    sess.run(init)
    quelen = sess.run(q.size())
    for i in range(2):
        sess.run(q_inc) #执行2次操作，队列中的值变为0.3, 1.1, 1.2

    qulen = sess.run(q.size())
    for i in range(quelen):
        print(sess.run(q.dequeue())) #输出对列的值

#RandomShuffleQueue
#创建一个随机队列，队列最大长度为10， 出队后最小长度为2：
q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")
#然后开启一个绘画，执行10次入队操作，8次出队操作
sess = tf.Session()
for i in range(0, 10):
    sess.run(q.enqueue(i))
for i in range(0, 8):
    print(sess.run(q.dequeue()))
#阻断发生
#队列长度等于最小值，执行出队操作
#队列长度等于最大值，执行入队操作

#可以通过设置绘画在运行时的等待时间来解除阻断
run_options = tf.RunOptions(timeout_in_ms=10000) #等待10秒
try:
    sess.run(q.dequeue(), options=run_options)
except tf.errors.DeadlineExceededError:
    print('out of range')

"""
############################################################################
#队列管理器
#我们创建一个含有一个队列的图
q = tf.FIFOQueue(1000, "float")
counter = tf.Variable(0.0) #计数器
increment_op = tf.assign_add(counter, tf.constant(1.0)) #操作：给计数器加1
enqueue_op = q.enqueue(counter) #操作：计数器值加入队列
#创建一个队列管理器QueueRunner,用这俩个操作向队列q中添加元素。目前我们只使用一个线程
qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 1)
#启动一个会话，从队列管理器qr中创建线程
#主线程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    enqueue_threads = qr.create_threads(sess, start=True) #启动入队线程
    #主线程
    for i in range(10):
        print(sess.run(q.dequeue()))

###############################################################################
#QueueRunner
#tf.train.Coordinator 实现线程间的同步，
#线程和协调器
#主线程
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Coordinator:协调器，协调线程间的关系可以视为一种信号量，用来做同步
coord = tf.train.Coordinator()

#启动入队线程，协调器是线程的参数
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

coord.request_stop()#通知其他线程关闭
#主线程
for i in range(0, 10):
    try:
        print(sess.run(q.dequeue()))
    except tf.errord.OutOfRangeError:
        break
coord.join(enqueue_threads)#join操作等待其他线程结束，其他所有线程关闭后，这一函数才能返回


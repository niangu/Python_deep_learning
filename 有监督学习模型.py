import tensorflow as tf
#初始化变量和模型参数,定义训练闭环中的运算
def inference(X):
    #计算推断模型在数据X上的输出，并将结果返回
    print()
def loss(X, Y):
    #依据训练数据X及其期望输出Y计算损失
    print()
def inputs():
    #读取或生成训练数据X及其期望输出Y
    print()
def train(total_loss):
    #依据计算的总损失训练或调整模型参数
    print()
def evaluate(sess, X, Y):
    #对训练得到的模型进行评估
    print()

#在一个会话对象中启动数据流图，搭建流程
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #实际的训练迭代次数
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        #出于调试和学习的目的，查看损失在训练过程中递减的情况
        if step % 10 == 0:
            print("loss:", sess.run([total_loss]))
    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    
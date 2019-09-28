import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("variables"):
        #记录数据流图运行次数的Variable对象
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        #追踪该模型的所有输出随时间的累加和的Variable对象
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

    with tf.name_scope("transformation"):
        #独立离的输入层
        with tf.name_scope("input"):
            #创建输出占位符号,用于接收一个向量
            a = tf.placeholder(tf.float32, shape=[None], name="input_placeholder_a")
        #独立的中间层
        with tf.name_scope("intermediate_layer"):
            b = tf.reduce_prod(a, name="product_b")
            c = tf.reduce_sum(a, name="sum_c")
            #对于乘法和加法运算，这里并未使用tf.mul()和tf.add(),而是分别使用了tf.reduce_prod(), tf.reduce_sum()
            #这样便可以对整个输入向量实施乘法和加法运算，而之前的Op只能接收俩个标量作为输入
        #独立的输出层
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")

    with tf.name_scope("update"):
        #用最新的输出更新Variable对象total_output
        update_total = total_output.assign_add(output)

        #将前面的Variable对象global_step增1，只要数据流图运行，该操作便需要进行
        increment_step = global_step.assign_add(1)

    with tf.name_scope("summaries"):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")
        #为输出节点创建汇总数据
        #tf.summary.scalar(b'Output', output, name="output_summary")
        #tf.summary.scalar(b'Sum of outputs over time', update_total, name="total_summary")
        #tf.summary.scalar(b'Average of outputs over time', avg, name="average_summary")
        tf.summary.scalar("output_summary", output)
        tf.summary.scalar("total_summary", update_total)
        tf.summary.scalar("average_summary", avg)

    with tf.name_scope("global_ops"):
            #初始化Op
            init = tf.initialize_all_variables()
            #将所有汇总数据合并到一个Op中
            merged_summaries = tf.summary.merge_all()

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('/home/niangu/my_graph', graph=graph)
#初始化Variable对象
sess.run(init)

def run_graph(input_tensor):
    '''
    辅助函数；用于给定的输入张量运行数u流图，
    并保存汇总数据
    '''
    feed_dict = {a: input_tensor}
    _, step, summary = sess.run([output, increment_step, merged_summaries], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)

#用于不同的输入运行该数据流图
run_graph([2, 8])
run_graph([3, 1, 3, 3])
run_graph([8])
run_graph([1, 2, 3])
run_graph([11, 4])
run_graph([4, 1])
run_graph([7, 3, 1])
run_graph([6, 3])
run_graph([0, 2])
run_graph([4, 5, 6])

#将数据汇总写入磁盘
writer.flush()
#关闭SummaryWriter对象
writer.close()
#关闭Session对象
sess.close()
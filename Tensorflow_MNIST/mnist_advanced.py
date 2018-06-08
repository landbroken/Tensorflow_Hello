# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# #### 1.设置输入和输出节点的个数,配置神经网络的参数。

# In[2]:


# MNIST数据集相关的常数
INPUT_NODE = 784  # 输入节点个数，因为28x28=784
OUTPUT_NODE = 10  # 输出节点个数，因为（0-9）10个数字

# 配置神经网络的参数
LAYER1_NODE = 500  # 这里使用隐藏层数只有一个的网络结构，而节点有500个
BATCH_SIZE = 100  # 每次batch打包的样本个数，个数越小训练过程越接近随机梯度下降，数字越大，训练越接近梯度下降

# 模型相关的参数
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化在损失函数中的系数
TRAINING_STEPS = 5000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# #### 2. 定义辅助函数来计算前向传播结果，使用ReLU做为激活函数。

# In[3]:


# 辅助函数给定神经网络的输入和所有参数，计算前向传播结果。在这里是一个三层的全连接神经网络，RELU函数可以去线性化，同时也可以传入用于计算平均值的类，
# 这样方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 不使用滑动平均类时，直接使用参数当前的取值
    if avg_class is None:
        # 计算隐藏层的前向传播结果，这里使用RELU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 使用滑动平均类：首先使用avg_class.average函数来计算得出变量的滑动平均值，然后再计算相应的前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# #### 3. 定义训练过程。

# In[4]:


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数。
    # 这里是784个输入节点，500个隐层接点，也就是784x500的矩阵
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))  # 偏置是根据隐层的节点数而定的

    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))  # 同上
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  # 输出层的节点的参数

    # 计算不含滑动平均类的前向传播结果，因为这里的avg_class=NONE，所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关的滑动平均类

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=false）,在tensorflow训练神经网络中
    # 一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    #  给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。这里知道给定训练轮数的变量可以加快训练早期变量的更新速度。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  # moving_average_decay
    # 在所有代表神经网络参数的变量上使用滑动平均，而其他的辅助变量就不需要了。tf.trainable_variables()返回的就是图上集合GraphKes.TRAINABLE_VARIABLES
    # 中的元素，这个集合的元素就是所有没有指定trainable=false的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用了滑动平均之后的前向传播结果。但滑动平均不会改变变量本身的值，而是会维护一个影子变量来记录其滑动平均值。所以需要使用滑动平均值时
    # 就需要明确调用average函数
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值：其中交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了tensorflow提供的tf.nn.sparse_softmax_cross_entropy_with_logits
    # 来计算交叉熵。第一个参数是神经网络不包括softmax层的前向传播结果，第二个是给定的训练数据的正确答案。argmax得到的是正确答案对应的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算在当前batch中 所有样例的交叉熵平均值

    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  # 计算L2正则化损失函数
    # 计算模型的正则化损失函数。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularaztion = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,  # 当前迭代的轮数，初始值为0
        mnist.train.num_examples / BATCH_SIZE,  # 跑完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY,  # 学习率衰减速度
        staircase=True)  # 决定衰减学习率的曲线图是何种形式，这里是阶梯衰减

    # 优化损失函数：这里使用tf.train.GradientDescentOptimizer优化算法来优化损失函数，注意这里的损失函数包括了交叉熵函数和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    # 在训练神经网络时，每过一遍数据既需要通过反向传播更新神经网络的参数，又需要更新每一个参数的滑动平均值，为了一次完成多个操作，
    # tensorflow提供了 tf.control_dependencies和tf.group两种机制。
    with tf.control_dependencies(
            [train_step, variables_averages_op]):  # 等同于train_op = tf.group（train_step, variables_averages_op)
        train_op = tf.no_op(name='train')

    # 计算正确率：
    # 检查使用了滑动平均模型的神经网络前向传播结果是否正确。
    # tf.argmax(average_y, 1)计算每一个样例的预测答案。其中average_y是一个batch*10的二维数组，每一行表示一个样例的前向传播结果。
    # 第二个参数1表示选取最大值的操作仅在第一个维度中进行（也就是说只在每一行中选取最大值的下标）。
    # 于是得到的结果是一个长度为batch的一维数组，这个一维数组中的值就表示了每一个样例对应的数字识别结果。
    # tf.equal判断两个张量的每一维是否相等，如果相等则返回TRUE，否则返回False
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))  # 简单来说就是判断预测结果和真实结果是否相同
    # 这里首先将布尔值转换为实数型，然后再计算平均值。这个平均值就是模型在这一维数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据。在真实的应用中，这部分数据在训练的时候是不可见的，这个数据只是作为模型优劣的最后评判标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:  # 每1000轮输出一次在验证数据集上的测试结果
                # 计算滑动平均模型在验证数据上的结果。这里由于MNIST数据集比较小，所以一次可以处理所有的验证数据。但如果是太大的数据集不化分为小的
                # batch会导致计算时间过长甚至发生内存溢出
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

            # 在训练结束之后，在测试数据集上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))


# #### 4. 主程序入口，这里设定模型训练次数为5000次。

# In[5]:


def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载（当然要联网），但这里我使用已经下载好的数据集
    MNIST_data_folder = r"G:\dataFile\minist"
    mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)
    print("Training data size: ", mnist.train.num_examples)  # 打印训练数据集大小
    print("Validating data size: ", mnist.validation.num_examples)  # 打印验证数据集的大小
    print("Testing data size: ", mnist.test.num_examples)  # 打印测试数据集的大小
    train(mnist)


if __name__ == '__main__':
    main()

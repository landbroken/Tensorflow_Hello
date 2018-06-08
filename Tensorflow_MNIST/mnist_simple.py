import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def train(mnist):
    # ***************
    # 实现回归模型
    # ***************
    # 我们通过操作符号变量来描述这些可交互的操作单元
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # ***************
    # 训练模型
    # ***************
    # 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：
    y_ = tf.placeholder("float", [None, 10])
    # 计算交叉熵:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 初始化我们创建的变量
    init = tf.initialize_all_variables()
    # 现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
    sess = tf.Session()
    sess.run(init)
    # 然后开始训练模型，这里我们让模型循环训练1000次！
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # ***************
    # 评估我们的模型
    # ***************
    # 找出那些预测正确的标签
    # tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
    # 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 计算所学习到的模型在测试数据集上面的正确率。
    output = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(output)


def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载（当然要联网），但这里我使用已经下载好的数据集
    MNIST_data_folder = r"G:\dataFile\minist"
    # 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。
    # 比如，标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])
    mnist = input_data.read_data_sets(MNIST_data_folder, one_hot=True)
    print("Training data size: ", mnist.train.num_examples)  # 打印训练数据集大小
    print("Validating data size: ", mnist.validation.num_examples)  # 打印验证数据集的大小
    print("Testing data size: ", mnist.test.num_examples)  # 打印测试数据集的大小
    train(mnist)


if __name__ == '__main__':
    main()

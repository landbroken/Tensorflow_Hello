import tensorflow as tf

# 标记的方法是使用 tf.placeholder() 为这些操作创建占位符.
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run([output], feed_dict={input1: [7.], input2: [2.]})
    print(result)

    # 输出:
    # [array([ 14.], dtype=float32)]

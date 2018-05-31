# 矩阵乘法
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
# result = sess.run(product)
# print(result)
# # 任务完成, 需要显式关闭会话，释放资源.
# sess.close()

# 用with自动关闭会话
with tf.Session() as sess:
    result = sess.run([product])
    print(result)

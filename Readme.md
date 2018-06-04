# 引言
这是一个TensorFlow的基本用法入门版  
主要参考了[中文社区]
(http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html)
的新手入门部分  

使用的是Python 3.6.3 + Tensorflow 1.8.0，所以和参考文档中的略有不同
# 文件简介
1、hello.py是第一个程序，日常hello world；  
2、matmul.py是第二个程序，简单的矩阵乘法；  
3、InteractiveSession是第三个程序，交互式使用；  
由于ipython安装有点问题，目前不能用，代码没问题，用.py版验证过了
4、variables.py演示了如何使用变量实现一个简单的计数器；    
5、fetch.py，在之前的例子里, 我们只取回了单个节点 state, 
但是你也可以取回多个 tensor；  
6、feed,py，TensorFlow 还提供了 feed 机制, 
该机制 可以临时替代图中的任意操作中的 tensor 
可以对图中任何操作提交补丁, 直接插入一个 tensor.  
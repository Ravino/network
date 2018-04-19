# encoding: utf-8

import glob as gb
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import read, write


dbClass = {0 : "seven", 1 : "house"}

sevenTrain = gb. glob ("./train/seven/*.wav");
houseTrain = gb. glob ("./train/house/*.wav");

sevenTest = gb. glob ("./test/seven/*.wav");
houseTest = gb. glob ("./test/house/*.wav");


maxSize = 16000;
for name in sevenTrain:
  tmp = read (name) [1];

  if (len (tmp) > maxSize):
    maxSize = len (tmp);


xData = np. zeros ((len (sevenTrain), maxSize), dtype = "float32");
yData = np. zeros ((len (sevenTrain) + len (houseTrain), 2), dtype = "float32");

for i, name in enumerate (sevenTrain):
  count1, sevenData = read (name);
  size = len (sevenData);
  xData [i, 0 : size] = sevenData;

  yData [i, : ] = np. array ([1, 0]);



xData1 = np. zeros ((len (houseTrain), maxSize), dtype = "float32");

for i, name in enumerate (houseTrain):
  count1, houseData = read (name);
  size = len (houseData);
  xData1 [i, 0 : size] = houseData;

  yData [i, :] = np. array ([0, 1]);

xData = np. vstack ([xData, xData1]);



def conv2d(x, W):
    # strides: по умолчанию порядок [Примеры, Вертикальный шаг, Горизонтальный шак, каналы]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv_layer(input, shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[shape[3]]))
    return tf.nn.relu(conv2d(input, W) + b)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = tf.Variable(tf.truncated_normal([in_size, size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size]))
    return tf.matmul(input, W) + b


# Входной массив
x = tf.placeholder(tf.float32, shape=[None, maxSize])

# Ответы
y_ = tf.placeholder(tf.float32, shape=[None, 2])


# Нужно переформатировать входы под ожидаемый формат входных данных
assert (500*32 == maxSize)
x_image = tf.reshape(x, [-1, 32, 500, 1])

# Первый сверточный слой
conv1 = conv_layer(x_image, shape=[1, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

# Второй сверточный слой
conv2 = conv_layer(conv1_pool, shape=[1, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

# Чтобы закинуть данные в полносвязную сеть, нужно сделать их "плоскими"
conv2_flat = tf.reshape(conv2_pool, [-1, 8*125*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

# Выходной слой
y_conv = full_layer(full_1, 2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predict = tf. argmax (y_conv, 1);

sess =  tf.Session()
sess.run(tf.global_variables_initializer())


STEPS = 100

for i in range(STEPS):
    sess.run(train_step, feed_dict={x: xData, y_: yData})
    if i % 10 == 0:
        valid_accuracy, answer = sess.run([accuracy, predict], 
                                  feed_dict={x: xData, y_: yData})
        print("step {}, точность {}".format(i, valid_accuracy))
#        print (answer)





#sess. run (predict, feed_dict = {x : xData});


print (dbClass [answer [0]]);


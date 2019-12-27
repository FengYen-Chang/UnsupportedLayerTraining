import tensorflow as tf 
import tensorflow.contrib.layers as layers

import numpy as np

weights = {
    'wc1': tf.Variable(tf.truncated_normal([1, 1, 1, 32]))
    }

biases = {
    'bc1': tf.Variable(tf.zeros([32]))
    }

def model (inputs) :
    conv1 = tf.nn.conv2d(inputs, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME') + biases['bc1']
    out = tf.math.cosh(conv1)
    return out

x = tf.placeholder(tf.float32, (None, 32, 32, 1))

o = model(x)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

print (sess.run(o, {x : np.ones((1, 32, 32, 1))}))

from tensorflow.python.framework import graph_io
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Cosh'])
graph_io.write_graph(frozen, '.', 'cosh.pb', as_text=False)



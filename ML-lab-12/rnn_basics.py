import numpy as np
import tensorflow as tf
import pprint

hidden_size = 2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[[1,0,0,0]]], dtype=np.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(outputs.eval())

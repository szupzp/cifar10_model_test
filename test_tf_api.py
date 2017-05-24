import tensorflow as tf
from  tensorflow.contrib import layers
data_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/train/*.JPEG'
list = tf.train.match_filenames_once(data_dir)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
dir_list = sess.run(list)
stop = 1
layers.apply_regularization()
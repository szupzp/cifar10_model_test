# By @Kevin Xu
# kevin28520@gmail.com
# Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
# The aim of this project is to use TensorFlow to process our own data.
#    - input_data_processing.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note:
# it is suggested to restart your kenel to train the model multiple times
#### (in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


# %%

import os
import numpy as np
import tensorflow as tf
import input_data_processing
import cifi_10_model as model
import test_on_one_image
from skimage import io
# %%

N_CLASSES = 2
IMG_W = 224  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 224
RATIO = 0.2  # take 20% of dataset as validation data
BATCH_SIZE = 64
CAPACITY = 20000
MAX_STEP = 50000  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.00001  # with current parameters, it is suggested to use learning rate<0.0001

# %%
train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/train/'
test_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/test/'
logs_train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/logs/train/'
logs_val_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/logs/val/'
logs_test_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/logs/test/'
def training():
    # you need to change the directories to yours.


    train, train_label, val, val_label = input_data_processing.get_train_files_list(train_dir, RATIO)
    train_batch, train_label_batch = input_data_processing.get_batch(train,
                                                                     train_label,
                                                                     IMG_W,
                                                                     IMG_H,
                                                                     BATCH_SIZE,
                                                                     CAPACITY)
    val_batch, val_label_batch = input_data_processing.get_batch(val,
                                                                 val_label,
                                                                 IMG_W,
                                                                 IMG_H,
                                                                 BATCH_SIZE,
                                                                 CAPACITY)
    test, test_label = input_data_processing.get_test_files_list(test_dir)
    test_batch, test_label_batch = input_data_processing.get_batch(test,
                                                                     test_label,
                                                                     IMG_W,
                                                                     IMG_H,
                                                                     BATCH_SIZE,
                                                                     CAPACITY)
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
    # x_t = tf.placeholder(tf.float32, shape=[400, IMG_W, IMG_H, 3])
    # y_t = tf.placeholder(tf.float32, shape=[400, N_CLASSES])

    logits = model.inference(x, BATCH_SIZE, N_CLASSES)
    loss = model.losses(logits, y_)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits,y_)



    #def training_model():
    ####training
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_op = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
        test_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x: tra_images, y_: tra_labels})
                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op,feed_dict={x: tra_images, y_: tra_labels})
                    train_writer.add_summary(summary_str, step)

                if step % 200 == 0 or (step + 1) == MAX_STEP:

                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op,feed_dict={x: val_images, y_: val_labels})
                    val_writer.add_summary(summary_str, step)
                    ##test
                    test_images, test_labels = sess.run([test_batch, test_label_batch])
                    test_loss, test_acc = sess.run([loss, acc],
                                                   feed_dict={x: test_images, y_: test_labels})
                    print('** Step %d,test loss = %.2f, test accuracy = %.2f%%  **' % (step, test_loss, test_acc * 100.0))
                    summary_str = sess.run(summary_op,feed_dict={x: test_images, y_: test_labels})
                    test_writer.add_summary(summary_str, step)
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train, train_label):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    print ('the file name is : %s \nreally label is : %d' %(train[ind],train_label[ind ,1]))
    image = io.imread(img_dir)
    io.imshow(image)
    image = np.array(image)
    return image

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/test/'
   train, train_label = input_data_processing.get_test_files_list(train_dir)
   image_array = get_one_image(train,train_label)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 224, 224, 3])

       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[224, 224, 3])

       # you need to change the directories to yours.
       logs_train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/logs/train/'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a Normal with possibility %.6f' %prediction[0, 0])
           else:
               print('This is a Pataology with possibility %.6f' %prediction[0, 1])
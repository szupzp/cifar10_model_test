from PIL import Image
from skimage import io
import  tensorflow as tf
import  numpy as np
import  cifi_10_model as model

from skimage import io
def get_one_image(val, val_lables):
    '''Randomly pick one image from training data
   Return: ndarray
   '''
    n = len(val)
    ind = np.random.randint(0, n)
    img_dir = val[ind]

    image = Image.open(img_dir)
    io.imshow(image)
    if val_lables[ind] == 0:
        print('this is normal example with lable:%d' % val_lables[ind])
    if val_lables[ind] == 1:
        print('this is pathology example with lable:%d' % val_lables[ind])
    
    # image = image.resize([208, 208])
    image = np.array(image)
    return image


def evaluate_one_image(test,test_label):
    '''Test one image against the saved models and parameters
   '''

    # you need to change the directories to yours.
    # test_dir = '/home/program/PycharmProjects/pzp_vgg16_project/data/test/'
    # test, test_label = input_data_processing.get_test_files_list(test_dir)


    image_array = get_one_image(test, val_lables=test_label)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        logit = model.inference(image_array, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[ 224, 224, 3])

        # you need to change the directories to yours.
        # logs_train_dir = '/home/program/PycharmProjects/pzp_vgg16_project/logs/train/'

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is Normal with possibility %.6f' % prediction[0, 0])
            else:
                print('This is Pathology with possibility %.6f' % prediction[0, 1])

import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 208
IMG_H = 208

train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
   i = 0
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)

   try:
       while not coord.should_stop() and i<1:

           img, label = sess.run([image_batch, label_batch])

           # just test one batch
           for j in np.arange(BATCH_SIZE):
               print('label: %d' %label[j])
               plt.imshow(img[j,:,:,:])
               plt.show()
           i+=1

   except tf.errors.OutOfRangeError:
       print('done!')
   finally:
       coord.request_stop()
   coord.join(threads)


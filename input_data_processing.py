# %%

import tensorflow as tf
import numpy as np
import os
import math

# %%



def get_train_files_list(file_dir, ratio = 0.2):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    class0 = []
    label_class0 = []
    class1 = []
    label_class1 = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if 'N' in name[0]:
            class0.append(file_dir + file)           #class 0 is normal
            label_class0.append(0)
        if 'P' in name[0]:
            class1.append(file_dir + file)           #class 1 is pathology
            label_class1.append(1)
    print('There are %d normal and\nThere are %d pathology in train set' % (len(class0), len(class1)))

    image_list = np.hstack((class0, class1))
    label_list = np.hstack((label_class0, label_class1))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)   
    
    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]
    
    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample*ratio) # number of validation samples
    n_train = int(n_sample - n_val) # number of trainning samples
    
    tra_images = all_image_list[0 : n_train]
    tra_labels = all_label_list[0 : n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train: -1]
    val_labels = all_label_list[n_train: -1]
    val_labels = [int(float(i)) for i in val_labels]
    
    
    
    return tra_images,tra_labels,val_images,val_labels


def get_test_files_list(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    class0 = []
    label_class0 = []
    class1 = []
    label_class1 = []
    for file in os.listdir(file_dir):
        name = file.split('.')
        if 'N' in name[0]:
            class0.append(file_dir + file)  # class 0 is normal
            label_class0.append(0)
        if 'P' in name[0]:
            class1.append(file_dir + file)  # class 1 is pathology
            label_class1.append(1)
    print('There are %d normal and\nThere are %d pathology in test set' % (len(class0), len(class1)))

    image_list = np.hstack((class0, class1))
    label_list = np.hstack((label_class0, label_class1))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# %%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch




# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 224
# IMG_H = 224
#
# train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/train/'
# ratio = 0.2
# tra_images, tra_labels, val_images, val_labels = get_files(train_dir, ratio)
# tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([tra_image_batch, tra_label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)

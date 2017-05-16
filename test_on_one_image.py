from PIL import Image
import matplotlib.pyplot as plt
import  input_data_processing
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
    # io.imshow(image)
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
import  input_data_processing
import  tensorflow as tf
import  numpy as np
import  cifi_10_model as model
from skimage import io

def read_test_set(image1, label1,batch_size,image_W, image_H,capacity):
    image = tf.cast(image1, tf.string)
    label = tf.cast(label1, tf.float32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size,2])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


'''Test one image against the saved models and parameters
   '''
BATCH_SIZE = 400
N_CLASSES = 2
CAPACITY = 15000
IMG_W = 224 
IMG_H = 224

# you need to change the directories to yours.
test_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/data_256/test/'
logs_train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/logs/train/'
test, test_label = input_data_processing.get_test_files_list(test_dir)
test_batch,test_label_batch = input_data_processing.get_batch(test, test_label, 224, 224,BATCH_SIZE,CAPACITY)
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE,2])
logits = model.inference(x, BATCH_SIZE, N_CLASSES)
prodiction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(prodiction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#acc = model.evaluation(logits, test_label_batch)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        tes_images, tes_labels = sess.run([test_batch, test_label_batch])
        test_acc = sess.run(accuracy,feed_dict={x: tes_images, y_: tes_labels})
        print('the test accuracy is: %.2f' % test_acc)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)

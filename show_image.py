import matplotlib.pyplot as plt

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 224
IMG_H = 224

train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/train/'
ratio = 0.2
tra_images, tra_labels, val_images, val_labels = get_tfiles(train_dir, ratio)
tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)



with tf.Session() as sess:
   i = 0
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)

   try:
       while not coord.should_stop() and i<1:

           img, label = sess.run([tra_image_batch, tra_label_batch])

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
# coding:utf-8
import os  # 处理字符串路径
import glob  # 查找文件

from keras.models import Sequential,Model  # 导入Sequential模型

from skimage.exposure import equalize_adapthist

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.applications.vgg16 import VGG16
train_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/data_256/train/'
test_dir = '/home/pzp/PycharmProjects/pzp_vgg16_project/data/data_256/test/'
base_model = VGG16(include_top=True, weights='imagenet')
# model = Sequential()
#
# model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))
x = base_model.get_layer('block5_pool').output
x = Flatten(name='flatten')(x)
x = Dense(1024, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
prodietion = Dense(2,activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prodietion)
for layer in base_model.layers:
    layer.trainble = False
adam = Adam(lr=0.0001)  # 采用随机梯度下降法，学习率初始值0.1,动量参数为0.9,学习率衰减值为1e-6,确定使用Nesterov动量
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])  # 配置模型学习过程，目标函数为categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    rotation_range=4,
#                                    fill_mode='constant',
#                                    cval=0,
#                                    horizontal_flip=True,
#                                    vertical_flip=True)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, color_mode='rgb',target_size=(224,224),batch_size=16)
test_generator = test_datagen.flow_from_directory(test_dir, color_mode='rgb',target_size=(224,224),batch_size=16)

tb = TensorBoard(log_dir='./logs/batch_size16/', write_images=True, histogram_freq=0)
checkpointer = ModelCheckpoint(filepath='./model/batch_size16.hdf5',save_best_only=True,verbose=1)

model.fit_generator(train_generator,steps_per_epoch=1000, epochs=70, workers=8, verbose=1,
                    validation_data=test_generator, validation_steps=360,callbacks=[tb,checkpointer])
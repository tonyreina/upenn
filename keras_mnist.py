from tensorflow import keras as K
import math
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epochs",
                    type=int,
                    default=3,
                    help="Number of epochs")
args = parser.parse_args()

print("TensorFlow version {}".format(tf.version.VERSION))
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("Intel DNNL enabled:", _pywrap_util_port.IsMklEnabled())
else:
   print("Intel DNNL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

batch_size = 128
num_classes = 10

epochs = args.epochs

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = K.utils.to_categorical(y_train, num_classes)
y_test = K.utils.to_categorical(y_test, num_classes)

model = K.models.Sequential()
model.add(K.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(K.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(K.layers.Dropout(0.25))
model.add(K.layers.Flatten())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.Dense(num_classes, activation='softmax'))

# Horovod: adjust learning rate based on number of GPUs.
opt = K.optimizers.Adadelta(1.0)


model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

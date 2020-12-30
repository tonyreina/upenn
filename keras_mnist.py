"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from tensorflow import keras as K
import math
import tensorflow as tf

import argparse

parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epochs",
                    type=int,
                    default=3,
                    help="Number of epochs")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size")
args = parser.parse_args()

print("TensorFlow version {}".format(tf.version.VERSION))
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("Intel DNNL enabled:", _pywrap_util_port.IsMklEnabled())
else:
   print("Intel DNNL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

num_classes = 10

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

# Define the model
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

# Horovod: adjust learning rate based on number of workers.
opt = K.optimizers.Adam(0.0001)

model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

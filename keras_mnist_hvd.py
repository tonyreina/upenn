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
import tensorflow as tf
import horovod.tensorflow.keras as hvd

import socket
import numpy as np

import argparse

parser = argparse.ArgumentParser(add_help=True, 
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--epochs",
                    type=int,
                    default=12,
                    help="Number of epochs")
parser.add_argument("--batch_size",
                    type=int,
                    default=128,
                    help="Batch size")
args = parser.parse_args()


# Horovod: initialize Horovod.
hvd.init()

num_classes = 10

print("Horovod size = {}, Horovod rank/worker # = {}, Hostname = {}".format(hvd.size(), hvd.rank(), socket.gethostname()))
rank_txt = "[Worker {} of {}]: ".format(hvd.rank(), hvd.size())

print("{} TensorFlow version {}".format(rank_txt, tf.version.VERSION))
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("{} Intel DNNL {}.".format(rank_txt, _pywrap_util_port.IsMklEnabled()))
else:
   print("{} Intel DNNL {}.".format(rank_txt, tf.pywrap_tensorflow.IsMklEnabled()))

# Load the dataset
####

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

####

# Preprocess dataset
####
# Shard the training dataset randomly based on the Horovod rank
num_shard = int(x_train.shape[0]//hvd.size())
x_train = x_train[hvd.rank()::hvd.size()] 
y_train = y_train[hvd.rank()::hvd.size()]
x_train = x_train[:num_shard]  # Make sure every worker has same sized shard
y_train = y_train[:num_shard]  # Make sure every worker has same sized shard

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("{} Train shape: {}".format(rank_txt, x_train.shape))
print("{} Test  shape: {}".format(rank_txt, x_test.shape))

# Convert class vectors to binary class matrices
y_train = K.utils.to_categorical(y_train, num_classes)
y_test = K.utils.to_categorical(y_test, num_classes)

####

# Define the model
####

inputs = K.layers.Input(input_shape, name="mnist")

conv1 = K.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu')(inputs)
conv2 = K.layers.Conv2D(64, (3, 3), activation='relu')(conv1)

maxpool1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
dropout1 = K.layers.Dropout(0.25)(maxpool1)
flat1 = K.layers.Flatten()(dropout1)
dense1 = K.layers.Dense(128, activation='relu')(flat1)
dropout2 = K.layers.Dropout(0.5)(dense1)
prediction = K.layers.Dense(num_classes, activation='softmax')(dropout2)

model = K.models.Model(inputs=[inputs], outputs=[prediction], name="mnist_model")
####

# Compile the model
####

# Horovod: adjust learning rate based on number of workers.
opt = K.optimizers.Adam(0.0001 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt_hvd = hvd.DistributedOptimizer(opt)

model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=opt_hvd,
              metrics=['accuracy'])
####

# Train/Fit model
####
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(K.callbacks.ModelCheckpoint('./horovod/checkpoints/checkpoint-{epoch}'))

model.fit(x_train, y_train,
          batch_size=args.batch_size,
          callbacks=callbacks,
          epochs=args.epochs,
          verbose=1 if hvd.rank() == 0 else 0,
          validation_data=(x_test, y_test))

# Evaluate best model
####
if hvd.rank() == 0:
   score = model.evaluate(x_test, y_test, verbose=1)

   print('Test loss:', score[0])
   print('Test accuracy:', score[1])

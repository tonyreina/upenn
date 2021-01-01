#! /bin/bash

module load horovod/0.21.0

python keras_mnist_hvd.py --epochs 3 --batch_size 128


#! /bin/bash

module load horovod/0.21.0

OMP_NUM_THREADS=16 python keras_mnist_hvd.py --epochs 8 --batch_size 64


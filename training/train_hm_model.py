import sys
import os
import pandas
import re
import math
sys.path.append("..")
from hm_model import get_training_model
from ds_iterator import DataIterator
from ds_generator_client import DataGeneratorClient
from optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.applications.resnet50 import ResNet50

import keras.backend as K

batch_size = 10
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 #68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000
stages = 3

# True = start data generator client, False = use augmented dataset file (deprecated)
use_client_gen = True

WEIGHTS_BEST = "resnet_trconv_hm_weights.h5"
TRAINING_LOG = "training_hm_model.csv"
LOGS_DIR = "./logs"

def get_last_epoch():
    data = pandas.read_csv(TRAINING_LOG)
    return max(data['epoch'].values)


model = get_training_model(weight_decay)

# load previous weights or vgg19 if this is the first run
if os.path.exists(WEIGHTS_BEST):
    print("Loading the best weights...")

    model.load_weights(WEIGHTS_BEST)
    last_epoch = get_last_epoch() + 1
else:
    print("Loading resnet50 weights...")
    last_epoch = 0

    rn = ResNet50(include_top=False, weights='imagenet', pooling=None)
    lc = 0
    for layer in model.layers:
        try:
            rn_layer = rn.get_layer(layer.name)
            if type(rn_layer) is Conv2D:
                print "Loading weights for layer", layer.name
                layer.set_weights(rn_layer.get_weights())
                lc += 1
        except:
            print "Skipping Layer ", layer.name

    print "Done loading weights for %d resnet conv layers" % lc
# prepare generators

train_client = DataGeneratorClient(port=5555, host="localhost", hwm=160, batch_size=batch_size, with_pafs=False, stages=stages)
train_client.start()
train_di = train_client.gen()
# train_samples = 52597 # All train samples in the COCO dataset
train_samples = 2000

val_client = DataGeneratorClient(port=5556, host="localhost", hwm=160, batch_size=batch_size, with_pafs=False, stages=stages)
val_client.start()
val_di = val_client.gen()
# val_samples = 2645 # All validation samples in the COCO dataset
val_samples = 200

# setup lr multipliers for conv layers
lr_mult=dict()
for layer in model.layers:

    if isinstance(layer, Conv2D):
        # stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[bias_name] = 8

        # Resnet conv layers
        else:
           kernel_name = layer.weights[0].name
           bias_name = layer.weights[1].name
           lr_mult[kernel_name] = 1
           lr_mult[bias_name] = 2

# configure loss functions

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

losses = {}
for i in range(1, stages+1):
    losses["weight_stage%d_L2" % i] = eucl_loss

# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = train_samples // batch_size
def step_decay(epoch):
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch

    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))
    return lrate

# configure callbacks
lrate = LearningRateScheduler(step_decay)

checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)

callbacks_list = [lrate, checkpoint, csv_logger, tb]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)

# start training
model.compile(loss=losses, optimizer=multisgd, metrics=["accuracy"])

model.fit_generator(train_di,
                    steps_per_epoch=train_samples // batch_size,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    validation_data=val_di,
                    validation_steps=val_samples // batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )
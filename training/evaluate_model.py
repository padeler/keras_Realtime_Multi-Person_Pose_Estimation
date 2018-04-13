from tensorflow import keras
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

from hm_model import acc_norm

batch_size = 100
base_lr = 4e-5 # 2e-5
momentum = 0.9
weight_decay = 5e-4
lr_policy =  "step"
gamma = 0.333
stepsize = 136106 # 68053   // after each stepsize iterations update learning rate: lr=lr*gamma
max_iter = 200000 # 600000

stages = 3

WEIGHTS_BEST = "resnet_trconv_hm_weights.h5"


val_client = DataGeneratorClient(port=5556, host="localhost", hwm=160, batch_size=batch_size, with_pafs=False, stages=stages)
val_client.start()
val_di = val_client.gen()
# val_samples = 2645 # All validation samples in the COCO dataset
val_samples = 2000

model = get_training_model(weight_decay)

# load previous weights or vgg19 if this is the first run
if os.path.exists(WEIGHTS_BEST):
    print("Loading the best weights...")

    model.load_weights(WEIGHTS_BEST)
else:
    raise("Weights file %s not found " % WEIGHTS_BEST)



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


# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

losses = {}
for i in range(1, stages+1):
    losses["weight_stage%d_L2" % i] = eucl_loss

print "KERAS VERSION: ", keras.__version__

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)
# start training
model.compile(loss=losses, optimizer=multisgd, metrics=[acc_norm])
print "Running model.evaluate() on %d samples" % val_samples
results = model.evaluate_generator(val_di,
                        steps=val_samples // batch_size,
                        use_multiprocessing=False)


print "Done! Results: "

for m,v in zip(model.metrics_names,results):
    print m," ==> ", v
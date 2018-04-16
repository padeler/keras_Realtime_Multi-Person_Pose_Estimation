from tensorflow import keras
import sys
import os
import pandas
import re
import math
sys.path.append("..")


from ds_iterator import DataIterator
from ds_generator_client import DataGeneratorClient
from keras.optimizers import Adadelta
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.applications.resnet50 import ResNet50

import keras.backend as K

from hm_model import acc_norm

batch_size = 100
train_samples = 2000

WEIGHTS_BEST = "vnect_weights.h5"


val_client = DataGeneratorClient(port=5556, host="localhost", hwm=160, batch_size=batch_size, with_pafs=False, stages=1)
val_client.start()
val_di = val_client.gen()
val_samples = 2000


import vnect_model as md
model = md.get_training_model()

# load previous weights or vgg19 if this is the first run
if os.path.exists(WEIGHTS_BEST):
    print("Loading the best weights...")
    model.load_weights(WEIGHTS_BEST)
else:
    raise("Weights file %s not found " % WEIGHTS_BEST)


def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

losses = {}
losses["weight_block"] = eucl_loss


adadelta = Adadelta()
model.compile(loss=losses, optimizer=adadelta, metrics=[acc_norm])

print "Running model.evaluate() on %d samples" % val_samples
results = model.evaluate_generator(val_di,
                        steps=val_samples // batch_size,
                        use_multiprocessing=False)


print "Done! Results: "

for m,v in zip(model.metrics_names,results):
    print m," ==> ", v
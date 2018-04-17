import sys
import os
import pandas

sys.path.append("..")
from ds_generator_client import DataGeneratorClient
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.applications.mobilenet import MobileNet, DepthwiseConv2D
from keras.optimizers import Adadelta

import keras.backend as K
from hm_model import acc_norm
from mobnet_model import get_training_model

batch_size = 12
max_epochs  = 200000 # 600000

WEIGHTS_BEST = "mobnet_weights.h5"
TRAINING_LOG = "training_mobnet.csv"
LOGS_DIR = "./logs"


def get_last_epoch():
    data = pandas.read_csv(TRAINING_LOG)
    return max(data['epoch'].values)


model = get_training_model()

if os.path.exists(WEIGHTS_BEST):
    print("Loading the best weights...")
    model.load_weights(WEIGHTS_BEST)
    last_epoch = get_last_epoch() + 1
else:
    print("Loading Mobnet weights...")
    last_epoch = 0
    mn = MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet', pooling=None)

    lc = 0
    for layer in model.layers:
        try:
            mn_layer = mn.get_layer(layer.name)
            if type(mn_layer) is Conv2D or type(mn_layer) is DepthwiseConv2D:
                print "Loading weights for layer", layer.name
                layer.set_weights(mn_layer.get_weights())
                lc += 1
        except:
            print "Skipping Layer ", layer.name

    print "Done loading weights for %d mobilenet conv layers" % lc

# prepare generators
stages = 1
train_client = DataGeneratorClient(port=5555, host="localhost", hwm=160, batch_size=batch_size, with_pafs=False, stages=stages)
train_client.start()
train_di = train_client.gen()
train_samples = 2000  # 52597 # All train samples in the COCO dataset

val_client = DataGeneratorClient(port=5556, host="localhost", hwm=160, batch_size=batch_size, with_pafs=False, stages=stages)
val_client.start()
val_di = val_client.gen()
val_samples = 200  # 2645 # All validation samples in the COCO dataset

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

losses = {}
losses["weight_block"] = eucl_loss

# configure callbacks
checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)

callbacks_list = [checkpoint, csv_logger, tb]

adadelta = Adadelta()


model.compile(loss=losses, optimizer=adadelta, metrics=["accuracy", acc_norm])
model.fit_generator(train_di,
                    steps_per_epoch=train_samples // batch_size,
                    epochs=max_epochs,
                    callbacks=callbacks_list,
                    validation_data=val_di,
                    validation_steps=val_samples // batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )
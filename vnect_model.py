from tensorflow import keras
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Add, Multiply, LeakyReLU

from keras.layers.pooling import MaxPooling2D
import keras.backend as K

from keras.applications.resnet50 import conv_block, identity_block

if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

def resnet50_block(img_input):


    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x


def vnect_block1(input):
    # top branch
    x = Conv2D(512, (1, 1), padding='same', name="MConv1_block1")(input)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv11')(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name="MConv2_block1")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv21')(x)
    x = Activation('relu')(x)

    x = Conv2D(1024, (1, 1), padding='same', name="MConv3_block1")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv31')(x)
    # x = Activation('relu')(x) # XXX not in the original vnect

    # bottom branch
    x2 = Conv2D(1024, (1, 1), padding='same', name="MConv4_block1")(input)
    x2 = BatchNormalization(axis=bn_axis, name='bn_MConv41')(x2)
    # x = Activation('relu')(x) # XXX not in the original vnect


    # add top and bottom branch
    x = Add()([x, x2])
    x = LeakyReLU(0.01)(x) # LeakyReLU as per Vnect

    return x


def vnect_block2(x, hm_len, paf_len):

    x = Conv2D(256, (1, 1), padding='same', name="MConv1_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv12')(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name="MConv2_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv22')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (1, 1), padding='same', name="MConv3_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv32')(x)
    x = Activation('relu')(x)

    # top transposed convolution
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", name="MConv4_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_trconv42')(x)
    x = Activation('relu')(x)

    # final conv portion.
    x = Conv2D(128, (3, 3), padding='same', name="MConv5_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv52')(x)
    x = Activation('relu')(x)

    hm = Conv2D(hm_len, (1, 1), padding='same', name="conv_hm")(x)
    hm = BatchNormalization(axis=bn_axis, name='bn_hm')(hm) # XXX do we need bn at the final conv?
    hm = Activation('softmax', name="sm_hm")(hm) # XXX not in the original vnect

    pafs = Conv2D(paf_len, (1, 1), padding='same', name="conv_paf")(x)
    pafs = BatchNormalization(axis=bn_axis, name='bn_paf')(pafs) 

    return hm, pafs

def get_training_model():

    hm_len = 19
    paf_len = 38

    img_input_shape = (None, None, 3)
    hm_input_shape = (None, None, hm_len)
    paf_input_shape = (None, None, paf_len)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    hm_weight_input = Input(shape=hm_input_shape)
    paf_weight_input = Input(shape=paf_input_shape)

    inputs.append(img_input)
    inputs.append(paf_weight_input)
    inputs.append(hm_weight_input)

    # For TF backend: resnet50 expects image input in range [-1.0,1.0]
    img_normalized = Lambda(lambda x: x / 127.5 - 1.0)(img_input)

    # RESNET50 up to block 4f and a transposed convolution in the end to increase resolution
    stage0_out = resnet50_block(img_normalized)

    block1_out = vnect_block1(stage0_out) # up to the sum
    hm, pafs = vnect_block2(block1_out, hm_len, paf_len)

    hm_out = Multiply(name="weight_block_hm")([hm, hm_weight_input])
    paf_out = Multiply(name="weight_block_paf")([pafs, paf_weight_input])
    
    outputs.append(paf_out)
    outputs.append(hm_out)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_testing_model(img_input_shape = (None, None, 3)):
    hm_len = 19
    paf_len = 38

    img_input = Input(shape=img_input_shape)

    # For TF backend: resnet50 expects image input in range [-1.0,1.0]
    img_normalized = Lambda(lambda x: x / 127.5 - 1.0)(img_input)

    # RESNET50 up to block 4f and a transposed convolution in the end to increase resolution
    stage0_out = resnet50_block(img_normalized)

    block1_out = vnect_block1(stage0_out) # up to the sum
    hm, pafs = vnect_block2(block1_out, hm_len, paf_len)

    model = Model(inputs=[img_input], outputs=[pafs, hm])

    return model



if __name__ == "__main__":
    model = get_testing_model((368,368,3))
    model.summary()
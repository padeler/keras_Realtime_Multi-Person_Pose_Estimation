from tensorflow import keras
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Add, Multiply, LeakyReLU
from keras.applications import mobilenet

from keras.layers.pooling import MaxPooling2D
import keras.backend as K

from keras.applications.mobilenet import _conv_block, _depthwise_conv_block, relu6

if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1


def mobilenet_block(img_input, alpha=1.0, depth_multiplier=1):

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x_mid = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x_mid, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x_mid = _depthwise_conv_block(x_mid, 128, alpha, depth_multiplier, block_id=15)
    # XXX TODO Add a transpose conv here and concat with earlier features from block 5 to increase res.
    # x = Conv2DTranspose(512, (4, 4), strides=(2, 2), padding="same", name="MConv4_block2")(x)
    # x = BatchNormalization(axis=bn_axis, name='bn_trconv42')(x)
    # x = Activation('relu')(x)

    return x, x_mid


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


def vnect_block2(x, num_p):

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

    x = Conv2D(num_p, (1, 1), padding='same', name="MConv6_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv62')(x) # XXX do we need bn at the final conv?
    # x = Activation('relu', name="block2_out")(x) # XXX not in the original vnect

    return x


def LeakyReLU6(x):
    return K.relu(x, alpha=0.01, max_value=6)


def vnect_dwc_block1(input, alpha=1.0, depth_multiplier=1):
    # top branch
    x = Conv2D(512, (1, 1), use_bias=False, padding='same', name="MConv1_block1")(input)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv11')(x)
    x = Activation(relu6)(x)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=12)

    x = Conv2D(1024, (1, 1), use_bias=False, padding='same', name="MConv3_block1")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv31')(x)

    # bottom branch
    x2 = Conv2D(1024, (1, 1), use_bias=False, padding='same', name="MConv4_block1")(input)
    x2 = BatchNormalization(axis=bn_axis, name='bn_MConv41')(x2)


    # add top and bottom branch
    x = Add()([x, x2])
    # LeakyReLU6 (following vnect leakyReLu and relu6 from mobnet)
    x = Activation(LeakyReLU6)(x)


    return x


def vnect_dwc_block2(x, x_mid, num_p, alpha=1.0, depth_multiplier=1):

    x = Conv2D(256, (1, 1), use_bias=False, padding='same', name="MConv1_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv12')(x)
    x = Activation(relu6)(x)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=13)

    x = Conv2D(256, (1, 1), use_bias=False, padding='same', name="MConv3_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv32')(x)
    x = Activation(relu6)(x)

    # top transposed convolution
    x = Conv2DTranspose(128, (4, 4), use_bias=False, strides=(2, 2), padding="same", name="MConv4_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_trconv42')(x)
    x = Activation(relu6)(x)

    x = Concatenate()([x,x_mid])

    # final conv portion.
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=14)

    x = Conv2D(num_p, (1, 1), use_bias=False, padding='same', name="MConv6_block2")(x)
    x = BatchNormalization(axis=bn_axis, name='bn_MConv62')(x) # XXX do we need bn at the final conv?
    # x = Activation('relu', name="block2_out")(x) # XXX not in the original vnect

    return x

def get_training_model():

    np_branch2 = 19 # heatmaps 18 parts + background

    img_input_shape = (None, None, 3)
    heat_input_shape = (None, None, np_branch2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(heat_weight_input)

    # For TF backend: resnet50 expects image input in range [-1.0,1.0]
    img_normalized = Lambda(lambda x: x / 127.5 - 1.0)(img_input)

    # mobilenet up to block 11
    stage0_out, x_mid = mobilenet_block(img_normalized)

    block1_out = vnect_dwc_block1(stage0_out) # up to the sum
    block2_out = vnect_dwc_block2(block1_out, x_mid, np_branch2)

    tr_out = Multiply(name="weight_block")([block2_out, heat_weight_input])
    outputs.append(tr_out)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model(img_input_shape = (None, None, 3)):
    np_branch2 = 19 # Heatmaps

    img_input = Input(shape=img_input_shape)

    # For TF backend: resnet50 expects image input in range [-1.0,1.0]
    img_normalized = Lambda(lambda x: x / 127.5 - 1.0)(img_input)

    # mobnet up to block 4f and a transposed convolution in the end to increase resolution
    stage0_out, x_mid = mobilenet_block(img_normalized)

    block1_out = vnect_dwc_block1(stage0_out) # up to the sum
    block2_out = vnect_dwc_block2(block1_out, x_mid, np_branch2)

    model = Model(inputs=[img_input], outputs=[block2_out])

    return model




if __name__=="__main__":
    model = get_testing_model((368,368,3))
    model.summary()



from tensorflow import keras
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Add, Multiply
from keras.layers.merge import add
from keras.applications import mobilenet

from keras.layers.pooling import MaxPooling2D
import keras.backend as K
from mobilenets import _depthwise_conv_block_v2, _conv_block, relu6, DepthwiseConv2D


stages = 4
np_branch2 = 19 # heatmaps 18 parts + background

if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1


def LeakyReLU6(x):
    return K.relu(x, alpha=0.01, max_value=6)


def mobilenet2_block(img_input, alpha=1.0, expansion_factor=6, depth_multiplier=1):

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block_v2(x, 16, alpha, 1, depth_multiplier, block_id=1)

    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, block_id=2, strides=(2, 2))
    x = _depthwise_conv_block_v2(x, 24, alpha, expansion_factor, depth_multiplier, block_id=3)

    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=4, strides=(2, 2))
    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=5)
    x = _depthwise_conv_block_v2(x, 32, alpha, expansion_factor, depth_multiplier, block_id=6)

    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=7)#, strides=(2, 2))
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=8)
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=9)
    x = _depthwise_conv_block_v2(x, 64, alpha, expansion_factor, depth_multiplier, block_id=10)
    
    x = _depthwise_conv_block_v2(x, 96, alpha, expansion_factor, depth_multiplier, block_id=11)
    x = _depthwise_conv_block_v2(x, 96, alpha, expansion_factor, depth_multiplier, block_id=12)
    x = _depthwise_conv_block_v2(x, 96, alpha, expansion_factor, depth_multiplier, block_id=13)

    # x = _depthwise_conv_block_v2(x, 160, alpha, expansion_factor, depth_multiplier, block_id=14)#, strides=(2, 2))
    # x = _depthwise_conv_block_v2(x, 160, alpha, expansion_factor, depth_multiplier, block_id=15)
    # x = _depthwise_conv_block_v2(x, 160, alpha, expansion_factor, depth_multiplier, block_id=16)

    # x = _depthwise_conv_block_v2(x, 320, alpha, expansion_factor, depth_multiplier, block_id=17)

    # if alpha <= 1.0:
    #     penultimate_filters = 1280
    # else:
    #     penultimate_filters = int(1280 * alpha)

    # x = _conv_block(x, penultimate_filters, alpha=1.0, kernel=(1, 1), block_id=18)


    # non mobnet2 layer
    # x = Activation(LeakyReLU6, name="mobnet_out")(x)

    return x



def stage_conv(inputs, filters, kernel=(3, 3), conv_id=1, stage_id=1, use_bn=True):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=not use_bn,
               name='stage%d_conv%d' % (stage_id, conv_id))(inputs)
    if use_bn:
        x = BatchNormalization(axis=bn_axis, name='stage%d_conv%d_bn' % (stage_id,conv_id))(x)

    return x

def stageT(x, num_p, stage_id=0, use_bn=True, slim=False):
    
    if not slim:
        for i in range(3):
            x = stage_conv(x, 128, kernel=3, conv_id=i, stage_id=stage_id)
            x = Activation(relu6, name='stage%d_conv%d_relu' % (stage_id, i))(x)

        # PW conv 
        x = stage_conv(x, 128, kernel=1, conv_id=3, stage_id=stage_id)
        x = Activation(relu6, name='stage%d_conv%d_relu' % (stage_id, 3))(x)
        
    # PW conv to the number of joints
    x = stage_conv(x, num_p, kernel=1, conv_id=4, stage_id=stage_id, use_bn=use_bn)
    x = Activation(LeakyReLU6, name='stage%d_leaky_relu6'%(stage_id))(x) 

    return x

def get_training_model():

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
    mobnet_out = mobilenet2_block(img_normalized)

    # stage0
    hm = stageT(mobnet_out, np_branch2, stage_id=0, slim=True)

    pre_out = Multiply(name="s0")([hm,  heat_weight_input])
    outputs.append(pre_out)
    
    x = Concatenate()([mobnet_out, hm])

    for sn in range(1,stages):
        stageT_out = stageT(x, np_branch2, sn)
        tr_out = Multiply(name="s%d"%sn)([stageT_out, heat_weight_input])
        outputs.append(tr_out)

        if (sn < stages):
            x = Concatenate()([mobnet_out, stageT_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model(img_input_shape = (None, None, 3)):

    img_input = Input(shape=img_input_shape)

    # For TF backend: resnet50 expects image input in range [-1.0,1.0]
    img_normalized = Lambda(lambda x: x / 127.5 - 1.0)(img_input)

    # mobilenet up to block 11
    mobnet_out = mobilenet2_block(img_normalized)

    # pre-stage
    hm = stageT(mobnet_out, np_branch2, stage_id=0, slim=True)
    x = Concatenate()([mobnet_out, hm])

    for sn in range(1, stages):

        stageT_out = stageT(x, np_branch2, sn)

        if (sn < stages):
            x = Concatenate()([mobnet_out, stageT_out])

    model = Model(inputs=[img_input], outputs=[stageT_out])

    return model



if __name__ == "__main__":
    model = get_testing_model((368,368,3))
    model.summary()


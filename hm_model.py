from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant

from model import vgg_block, conv, relu, apply_mask

def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization()(x)

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization()(x)

    return x



def get_training_model(weight_decay):

    stages = 6
    np_branch2 = 19 # heatmaps 18 parts + background

    img_input_shape = (None, None, 3)
    heat_input_shape = (None, None, 19)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, None, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch2_out, stage0_out])

    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - confidence maps
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, None, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_testing_model():
    stages = 6
    np_branch2 = 19 # Heatmaps

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch2_out, stage0_out])

    # stage t >= 2
    for sn in range(2, stages + 1):
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch2_out])

    return model
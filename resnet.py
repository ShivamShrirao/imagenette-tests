import tensorflow as tf
import tensorflow.keras.layers as layers

from resblock import norm_act, conv_norm, BasicBlock, Bottleneck, AttnBottleneck

# https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/applications/resnet.py


class Resnet():
    def __init__(self, block, filters_per_stack=[64, 128, 256, 512], num_repeats=[3,4,6,3], strides=[1,2,2,2],
                 expansion=4, dp_rate=0, activation=tf.nn.relu, inputs=None, input_shape=(224, 224, 3),
                 num_classes=1024):
        self.block = block
        self.dp_rate = dp_rate
        self.activation = activation
        self.num_classes = num_classes
        self.expansion = expansion
        self.input_shape = input_shape
        if inputs is None:
            self.inputs = layers.Input(shape=input_shape)
        else:
            self.inputs = inputs

        assert len(filters_per_stack) == len(num_repeats) == len(strides)
        self.filters_per_stack = filters_per_stack
        self.num_repeats = num_repeats
        self.strides = strides
        o = self.build_model()
        model = tf.keras.Model(inputs=self.inputs, outputs=o)

    def build_model(self, include_top=True):
        x = self.inputs
        # stem
        x = conv_norm(x, 64, kernel_size=7, strides=2, activation=self.activation, do_norm_act=False)
        x = layers.MaxPool2D(2, strides=2, data_format="channels_first")(x)

        # body
        for i in range(len(self.filters_per_stack)):
            x = self.stack(x, self.block, self.filters_per_stack[i], self.strides[i], repeat=self.num_repeats[i],
                      dp_rate=self.dp_rate)

        x = norm_act(x)
        if not include_top:
            return x
        else:
            x = layers.GlobalAveragePooling2D(data_format="channels_first")(x)
            x = layers.Dense(self.num_classes)(x)
            x = layers.Softmax(axis=-1)(x)
            return x

    def stack(self, x, block, filters, stride1=2, dp_rate=0, repeat=3, suffix=1):
        i = 0
        x = block(x, filters, strides=stride1, activation=self.activation, 
                       expansion=self.expansion, dp_rate=dp_rate, make_model=False,
                       suffix=f"{suffix}_{i}")
        for i in range(1, repeat-1):
            x = block(x, filters, strides=1, activation=self.activation, 
                       expansion=self.expansion, dp_rate=dp_rate, make_model=False,
                       suffix=f"{suffix}_{i}")
        return x

    def preprocess_input(self):
        raise NotImplementedError


def RESNEXT():
    raise NotImplementedError


def Resnet18(inputs=None,
             input_shape=(3,224,224),
             num_classes=1024,
             dp_rate=0,
             activation=tf.nn.relu):

    return Resnet(BasicBlock, num_repeats=[2,2,2,2], inputs=inputs, input_shape=input_shape, num_classes=num_classes,
                  dp_rate=dp_rate, activation=activation)

def Resnet34(inputs=None,
             input_shape=(3,224,224),
             num_classes=1024,
             dp_rate=0,
             activation=tf.nn.relu):

    return Resnet(BasicBlock, num_repeats=[3,4,6,3], inputs=inputs, input_shape=input_shape, num_classes=num_classes,
                  dp_rate=dp_rate, activation=activation)

def Resnet50(inputs=None,
             input_shape=(3,224,224),
             num_classes=1024,
             dp_rate=0,
             activation=tf.nn.relu):

    return Resnet(Bottleneck, num_repeats=[3,4,6,3], inputs=inputs, input_shape=input_shape, num_classes=num_classes,
                  dp_rate=dp_rate, activation=activation)

def Resnet101(inputs=None,
             input_shape=(3,224,224),
             num_classes=1024,
             dp_rate=0,
             activation=tf.nn.relu):

    return Resnet(Bottleneck, num_repeats=[3,4,23,3], inputs=inputs, input_shape=input_shape, num_classes=num_classes,
                  dp_rate=dp_rate, activation=activation)
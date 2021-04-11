import tensorflow as tf
import tensorflow.keras.layers as layers

from attention import SqueezeAttention2D, MultiHeadAttention2D

def norm_act(x, activation=tf.nn.relu):
    x = layers.BatchNormalization(axis=1)(x)

    if activation is not None:
        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        x = layers.Activation(activation)(x)
    return x


def conv_norm(x, filters, kernel_size=3, strides=1, activation=tf.nn.relu, do_norm_act=True):

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                      use_bias=not do_norm_act, data_format="channels_first")(x)
    if do_norm_act:
        x = norm_act(x, activation=activation)
    return x


def BasicBlock(inp, filters, strides=1, activation=tf.nn.relu, dp_rate=0,
               suffix=1, *args, **kwargs):

    in_filters = inp.shape[-1]

    x = norm_act(inp, activation=activation)

    identity = inp
    if in_filters != filters:   # use conv_shortcut to increase the filters of identity.
        identity = conv_norm(x, filters, kernel_size=1, strides=1, activation=activation, do_norm_act=False)
    elif strides > 1:           # else just downsample
        identity = layers.MaxPool2D(data_format="channels_first")(inp)

    x = conv_norm(x, filters, kernel_size=3, activation=activation, strides=strides)
    x = conv_norm(x, filters, kernel_size=3, activation=activation, do_norm_act=False)

    if dp_rate:
        x = layers.Dropout(dp_rate)(x)

    x = layers.Add()([identity, x])

    return x


def Bottleneck(inp, filters, strides=1, activation=tf.nn.relu, expansion=4, dp_rate=0,
               groups=1, base_width=64, squeeze_reduce=0, suffix=1, *args, **kwargs):

    in_filters = inp.shape[1]
    out_filters = filters*expansion
    width = int(filters * (base_width / 64.)) * groups

    x = norm_act(inp, activation=activation)

    identity = inp
    conv_shortcut = False
    if in_filters != out_filters:   # use conv_shortcut to increase the filters of identity.
        identity = conv_norm(x, out_filters, kernel_size=1, strides=1, activation=activation, do_norm_act=False)
        conv_shortcut = True
    if strides > 1:
        mip = identity if conv_shortcut else inp
        identity = layers.MaxPool2D(data_format="channels_first")(mip)

    x = conv_norm(x, width, kernel_size=1, activation=activation)      # contract
    x = conv_norm(x, width, kernel_size=3, activation=activation, strides=strides)
    x = conv_norm(x, out_filters, kernel_size=1, activation=activation, do_norm_act=False) # expand

    if squeeze_reduce:
        x = SqueezeAttention2D(squeeze_reduce)(x)

    if dp_rate:
        x = layers.Dropout(dp_rate)(x)

    x = layers.Add()([identity, x])

    return x


def AttnBottleneck(inp, filters, strides=1, activation=tf.nn.relu, expansion=4,
                   dp_rate=0, suffix=1, self_attn=False, nheads=8, pos_emb=True,
                   frac_dk=0.5, frac_dv=0.25, down_attn=False, *args, **kwargs):

    in_filters = inp.shape[1]
    out_filters = filters*expansion

    x = norm_act(inp, activation=activation)

    identity = inp
    conv_shortcut = False
    if in_filters != out_filters:   # use conv_shortcut to increase the filters of identity.
        identity = conv_norm(x, out_filters, kernel_size=1, strides=1, activation=activation, do_norm_act=False)
        conv_shortcut = True
    if strides > 1:
        mip = identity if conv_shortcut else inp
        identity = layers.MaxPool2D(data_format="channels_first")(mip)

    x = conv_norm(x, filters, kernel_size=1, activation=activation)      # contract
    kq = None
    dk = int(filters * frac_dk)
    dv = int(filters * frac_dv)
    cf = filters - dv if self_attn else filters
    if cf:
        x_c = conv_norm(x, cf, kernel_size=3, activation=activation, strides=strides,
                        do_norm_act=False)

    if self_attn:
        if down_attn or strides > 1:            # reduce size to save space or if conv does downsample and not for deconv as it increases so done during restore.
            x = layers.AveragePooling2D(pool_size=(4,4), strides=2, padding='same', data_format="channels_first")(x)
        o, kq = MultiHeadAttention2D(x, kq, dk=dk, dv=dv, nheads=nheads, pos_emb=pos_emb)
        if down_attn:                           # restore if previously reduced or if deconv does upsample.
            o = layers.UpSampling2D(data_format="channels_first")(o)
        x_c = layers.Concatenate(axis=1)([o, x_c]) if cf else o

    x = norm_act(x_c, activation=activation)
    x = conv_norm(x, out_filters, kernel_size=1, activation=activation, do_norm_act=False) # expand

    if not self_attn:
        x = SqueezeAttention2D()(x)

    if dp_rate:
        x = layers.Dropout(dp_rate)(x)

    x = layers.Add()([identity, x])

    return x
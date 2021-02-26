import tensorflow as tf
import tensorflow.keras.layers as layers

from attention import SqueezeAttention2D, AddPositionalEmbeddings, MultiHeadAttention2D

def norm_act(x, gn_grps=8, activation=tf.nn.leaky_relu):
    norm = CONFIG.norm
    if norm == 'gn':
        # assert filters % gn_grps == 0
        filters = x.shape[1]        # channel_first
        if filters < gn_grps:
            gn_grps = filters
        x = tfa.layers.GroupNormalization(groups=gn_grps, axis=1)(x)
    elif norm == 'bn':
        x = layers.BatchNormalization(axis=1)(x)

    if activation is not None:
        if activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        x = layers.Activation(activation)(x)
    return x


def conv_norm(x, filters, kernel_size=3, strides=1, gn_grps=8, activation=tf.nn.leaky_relu,
              do_norm_act=True):

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                      data_format="channels_first")(x)
    if do_norm_act:
        x = norm_act(x, gn_grps=gn_grps, activation=activation)
    return x


def block_conv_attn(inp, filters, strides=1, activation=tf.nn.leaky_relu,
                    conv_shortcut=False, self_attn=False, frac_dk=0.5,
                    squeeze_attn=True, nheads=8, down_attn=False, fscale=4,
                    dp_rate=0, make_model=False, pos_emb=True, frac_dv=0.25):

    if make_model:                  # to group block layers into a model
        svd_inp = inp
        inp = layers.Input(shape=inp.shape[1:])

    x = norm_act(inp, activation=activation)

    if conv_shortcut:               # use conv_shortcut to increase the filters of shortcut.
        shortcut = conv_norm(x, fscale*filters, 1, activation=activation, do_norm_act=False)
    elif strides > 1:               # else just downsample/upsample or keep the same.
        shortcut = layers.MaxPool2D(data_format="channels_first")(inp)
    else:
        shortcut = inp

    x = conv_norm(x, filters, kernel_size=1, activation=activation)      # contract
    kq = None
    dk = int(filters * frac_dk)
    dv = int(filters * frac_dv)
    cf = filters - dv if self_attn else filters
    x_s = conv_norm(x, cf, kernel_size=3, strides=strides, activation=activation,
                    do_norm_act=False)

    if self_attn:
        if down_attn or strides > 1:                               # reduce size to save space or if conv does downsample and not for deconv as it increases so done during restore.
            x = layers.AveragePooling2D(pool_size=(4,4), strides=2, padding='same', data_format="channels_first")(x)
        o, kq = MultiHeadAttention2D(x, kq, dk=dk, dv=dv, nheads=nheads, pos_emb=pos_emb)
        if (down_attn and strides < 2):                 # restore if previously reduced or if deconv does upsample.
            o = layers.UpSampling2D(data_format="channels_first")(o)
        x_s = layers.Concatenate(axis=1)([o, x_s])

    x = norm_act(x_s, activation=activation)
    x = conv_norm(x, fscale*filters, kernel_size=1, activation=activation, do_norm_act=False) # expand

    if squeeze_attn:
        x = SqueezeAttention2D()(x)

    if dp_rate:
        x = layers.Dropout(dp_rate)(x)

    x = layers.Add()([shortcut, x])
    if make_model:
        m = tf.keras.Model(inputs=inp, outputs=x)
        return m(svd_inp)
    else:
        return x

# decisive_depth
# x = block(args)(x)
# gp = globalpool(x)
# gamma = dense(1)(gp) # sigmoid or simply relu
# next_block = block(args)
# def call(args):         # or process in batch and mask with 0
#     x, next_block, gamma = args
#     if gamma > 0.5:
#         x = next_block(x)
#     else:
#         x = pool(x) (if stride>1 else x) or pointwise(x)
# x = tf.map_fn(call, (x, next_block, gamma))
# x = block(args)(x)

def down_stack(x, fltrs=[8,16,16], strides=2, self_attn=[False,False,False],
               fscales=[4,4,4], down_attn=False, squeeze_attn=True, dp_rate=0.2,
               fcnv_scut=False, activation=tf.nn.relu):
    if isinstance(self_attn, bool):
            self_attn = [self_attn]*3
    pos_emb = CONFIG.pos_emb
    frac_dk = CONFIG.frac_dk
    frac_dv = CONFIG.frac_dv

    x = block_conv_attn(x, fltrs[0], strides=strides, activation=activation, frac_dk=frac_dk,
                        conv_shortcut=fcnv_scut, pos_emb=pos_emb, squeeze_attn=squeeze_attn,
                        self_attn=self_attn[0], fscale=fscales[0], down_attn=down_attn,
                        dp_rate=dp_rate, make_model=False, frac_dv=frac_dv)

    x = block_conv_attn(x, fltrs[1], strides=1, activation=activation, frac_dk=frac_dk,
                        conv_shortcut=True, pos_emb=pos_emb, squeeze_attn=squeeze_attn,
                        self_attn=self_attn[1], fscale=fscales[1], down_attn=down_attn,
                        dp_rate=dp_rate, make_model=False, frac_dv=frac_dv)

    x = block_conv_attn(x, fltrs[2], strides=1, activation=activation, frac_dk=frac_dk,
                        conv_shortcut=True, pos_emb=pos_emb, squeeze_attn=squeeze_attn,
                        self_attn=self_attn[2], fscale=fscales[2], down_attn=down_attn,
                        dp_rate=dp_rate, make_model=False, frac_dv=frac_dv)
    return x
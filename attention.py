import tensorflow as tf
import tensorflow.keras.layers as layers

from utils import MatMulLayer

class SqueezeAttention2D(layers.Layer):
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gpool = layers.GlobalAveragePooling2D(data_format='channels_first')

    def build(self, input_shape):
        self.input_channels = input_shape[1]
        self.fc1 = layers.Dense(self.input_channels//self.ratio, activation='relu')
        self.fc2 = layers.Dense(self.input_channels, activation='hard_sigmoid')
        self.restore_shape = layers.Reshape(target_shape=(self.input_channels, 1, 1))

    def call(self, inp, training=False):
        x = self.gpool(inp)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.restore_shape(x)
        x = inp * x
        return x

    def cast_inputs(self, inp):
        return self._mixed_precision_policy.cast_to_lowest(inp)
    
    def get_config(self):
        l_config = {"ratio": self.ratio}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(l_config.items()))


class AddPositionalEmbeddings(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        kq, q_f = input_shape
        HW = kq[-1]
        dim_dk = q_f[-2]
        self.embedding = self.add_weight("pos_emb", shape=(1, 1, HW, dim_dk), initializer="truncated_normal", trainable=True)
    
    def call(self, inp):
        kq, q_f = inp
        return kq + tf.matmul(self.embedding, q_f)


def MultiHeadAttention2D(inp, prev_kq=None, dk=None, dv=None, nheads=8, pos_emb=True, name=""):   # inp [B, C, H, W]
    name_ = name+"_" if name else name
    dkh = dk//nheads            # channels per head
    dvh = dv//nheads
    inv_sqrt_dkh = dkh**-0.5
    kqv = layers.Conv2D(filters=2*dk + dv, kernel_size=1, strides=1, padding='same',
                        data_format='channels_first')(inp)  # inp [B, 2*dk+dv, H, W]
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=1)

    q = layers.Lambda(lambda x: tf.scalar_mul(inv_sqrt_dkh, x))(q)  # Done earlier cause q_f is smaller than kq so will be faster.
    flatten = lambda d_h: layers.Reshape((nheads, d_h, -1))
    k_f = flatten(dkh)(k)       # [B, N, dk/N, H*W]
    q_f = flatten(dkh)(q)       # [B, N, dk/N, H*W]
    v_f = flatten(dvh)(v)       # [B, N, dv/N, H*W]

    kq = MatMulLayer(transpose_a=True)([k_f, q_f])  # [B, N, H*W, H*W]          # or ADD ???
    if prev_kq is not None:
        kq = layers.Add()([kq, prev_kq])
    if pos_emb:
        kq = AddPositionalEmbeddings()([kq, q_f])   # [B, N, H*W, H*W]  # as in attention augmented convolution paper

    attn_map = layers.Softmax(axis=-1)(kq)          # [B, N, H*W, H*W]  # or TANH or SIGMOID ???

    o = MatMulLayer()([v_f, attn_map])              # [B, N, dv/N, H*W]
    o = layers.Reshape([dv]+inp.shape[-2:])(o)      # [B, dv, H, W]      # if reshaping gives error for None dim, use Lambda Layer and reshape in that.
    o = layers.Conv2D(filters=dv, kernel_size=1, strides=1, padding='same',
                      data_format='channels_first')(o)      # [B, dv, H, W]
    return o, kq
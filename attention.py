import tensorflow as tf
import tensorflow.keras.layers as layers


class SqueezeAttention2D(layers.Layer):
    def __init__(self, ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gpool = layers.GlobalAveragePooling3D(data_format='channels_first')

    def build(self, input_shape):
        self.input_channels = input_shape[1]
        print("input_channels:", input_channels)
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
        H, W = input_shape[-2:]
        self.embedding_H = self.add_weight("pos_H", shape=(1, 1, H, 1), initializer="truncated_normal", trainable=True)
        self.embedding_W = self.add_weight("pos_W", shape=(1, 1, 1, W), initializer="truncated_normal", trainable=True)
    
    def call(self, inp):
        return self.embedding_H + self.embedding_W + inp
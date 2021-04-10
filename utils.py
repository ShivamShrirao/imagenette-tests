import tensorflow as tf
import tensorflow.keras.layers as layers


class MatMulLayer(layers.Layer):
    def __init__(self, transpose_a=False, transpose_b=False, **kwargs):
        super().__init__(**kwargs)
        self.transpose_a=transpose_a
        self.transpose_b=transpose_b
    
    def get_config(self):
        l_config = {"transpose_a": self.transpose_a, "transpose_b": self.transpose_b}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(l_config.items()))

    def build(self, input_shape):
        assert len(input_shape)==2, "A `MatMulLayer` should be called on exactly 2 inputs"

    def call(self, inputs):
        x = tf.matmul(inputs[0], inputs[1], transpose_a=self.transpose_a, transpose_b=self.transpose_b)
        return x
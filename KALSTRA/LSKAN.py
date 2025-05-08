import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dense, LayerNormalization
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

class BlockDiagonal(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        assert out_features % num_blocks == 0
        block_out_features = out_features // num_blocks
        self.blocks = [tf.keras.layers.Dense(
            block_out_features) for _ in range(num_blocks)]

    def call(self, x):
        x = [block(x) for block in self.blocks]
        return tf.concat(x, axis=-1)

class mLSTMBlock(layers.Layer):
    def __init__(self, input_size, head_size, num_heads, proj_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.proj_factor = proj_factor
        assert proj_factor > 0

        self.layer_norm = LayerNormalization()
        self.up_proj = Dense(int(input_size * proj_factor))
        self.down_proj = Dense(input_size)
        self.Wq = BlockDiagonal(
            int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wk = BlockDiagonal(
            int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wv = BlockDiagonal(
            int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wi = Dense(self.hidden_size)
        self.Wf = Dense(self.hidden_size)
        self.Wo = Dense(self.hidden_size)
        self.group_norm = LayerNormalization()
        self.adjust_proj = Dense(self.hidden_size)

    def call(self, x):
        x_norm = self.layer_norm(x)
        x_up = self.up_proj(x_norm)
        x_up_adjusted = self.adjust_proj(x_up)
        q = self.Wq(x_up_adjusted)
        k = self.Wk(x_up_adjusted) / (self.head_size ** 0.5)
        v = self.Wv(x_up_adjusted)

        i_tilde = self.Wi(x_up_adjusted)
        f_tilde = self.Wf(x_up_adjusted)
        o = tf.sigmoid(self.Wo(x_up_adjusted))

        m_t = tf.maximum(f_tilde, i_tilde)
        i = tf.exp(i_tilde - m_t)
        f = tf.exp(f_tilde - m_t)
        c_t = f * v * k
        n_t = k

        a = o * (c_t * q)
        b = tf.maximum(tf.abs(n_t * q), 1)

        h_t = a / b
        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm * tf.sigmoid(x_up_adjusted)
        output = self.down_proj(output)
        final_output = output + x

        return final_output

class CFKANLayer(tf.keras.layers.Layer):
    def __init__(self, inputdim, outdim, gridsize, degree, addbias=True, smooth_initialization=False):
        super(CFKANLayer, self).__init__()
        self.gridsize = gridsize
        self.degree = degree
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        grid_norm_factor = (
            tf.range(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        self.cheby_coeffs = self.add_weight(
            shape=(outdim, inputdim, degree + 1),
            initializer=tf.random_normal_initializer(
                stddev=1.0 / (np.sqrt(inputdim) * grid_norm_factor)),
            trainable=True
        )

        if self.addbias:
            self.bias = self.add_weight(
                shape=(1, outdim),
                initializer=tf.zeros_initializer(),
                trainable=True
            )
        else:
            self.bias = None

    def call(self, x):
        xshp = tf.shape(x)
        outshape = tf.concat([xshp[:-1], [self.outdim]], axis=0)
        x = tf.reshape(x, (-1, self.inputdim))
        k = tf.reshape(tf.range(1, self.gridsize + 1,
                       dtype=x.dtype), (1, 1, 1, self.gridsize))
        x = tf.tanh(x)
        xrshp = tf.reshape(x, (tf.shape(x)[0], 1, tf.shape(x)[1], 1))

        cheby_polynomials = []
        for n in range(self.degree + 1):
            cos_theta = tf.acos(xrshp)
            cheby_polynomials.append(tf.cos(n * cos_theta))

        cheby_polynomials = tf.stack(cheby_polynomials, axis=-1)
        y = tf.reduce_sum(cheby_polynomials *
                          self.cheby_coeffs[None, :, :, :], axis=(-2, -1))

        if self.addbias:
            y += self.bias
        y = tf.reshape(y, outshape)
        return y

class ATLSTM_Block(layers.Layer):
    def __init__(self, model_size, num_heads, dff_size, maxlen, drop_path=0.1, **kwargs):
        super().__init__(**kwargs)
        self.filter = mLSTMBlock(
            model_size, model_size // num_heads, num_heads, proj_factor=2)
        self.drop_path1 = layers.Dropout(drop_path)
        self.drop_path2 = layers.Dropout(drop_path)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = CFKANLayer(model_size, model_size, 1, 4)

    def call(self, x, mask, training=True):
        x_filtered = self.filter(x)
        x_filtered = tf.cast(x_filtered, tf.float32)
        x = x_filtered + self.drop_path1(x_filtered, training=training)
        x = self.layernorm1(x)
        x = self.mlp(x)
        x = x + self.drop_path2(x, training=training)
        x = self.layernorm2(x)

        return x

def positional_embedding(maxlen,model_size):
    PE = np.zeros((maxlen,model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j%2 == 0:
                PE[i,j] = np.sin(i/10000**(j/model_size))
            else:
                PE[i,j] = np.cos(i/10000**((j-1)/model_size))
    PE = tf.constant(PE,dtype=tf.float32)
    return PE

class LSTMEncoder(keras.Model):
    def __init__(self, num_layers, model_size, num_heads, dff_size, vocab_size, maxlen, drop_path=0.1, **kwargs):
        super(LSTMEncoder, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size, model_size)
        self.pos_embedding = positional_embedding(maxlen, model_size)
        self.FFTencoder_layers = [ATLSTM_Block(
            model_size, num_heads, dff_size, maxlen, drop_path) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(drop_path)

    def call(self, x, training=True, padding_mask=None):
        x = self.embedding(x)+self.pos_embedding
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.FFTencoder_layers[i](x, padding_mask, training=True)
        return x

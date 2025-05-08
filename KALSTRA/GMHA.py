import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

class SpatialEncoding(tf.keras.layers.Layer):
    def __init__(self, max_path_distance: int):

        super(SpatialEncoding, self).__init__()
        self.max_path_distance = max_path_distance
        self.b = self.add_weight(shape=(self.max_path_distance,), initializer='random_normal', trainable=True)

    def call(self, x: tf.Tensor, paths) -> tf.Tensor:

        num_nodes = tf.shape(x)[0]
        spatial_matrix = tf.zeros((num_nodes, num_nodes), dtype=tf.float32)

        for src in paths:
            for dst in paths[src]:
                path_length = min(len(paths[src][dst]), self.max_path_distance) - 1
                spatial_matrix = tf.tensor_scatter_nd_update(
                    spatial_matrix, 
                    indices=[[src, dst]], 
                    updates=[self.b[path_length]]
                )

        return spatial_matrix

class EdgeEncoding(tf.keras.layers.Layer):
    def __init__(self, edge_dim: int, max_path_distance: int):

        super(EdgeEncoding, self).__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = self.add_weight(shape=(self.max_path_distance, self.edge_dim), initializer='random_normal', trainable=True)

    def call(self, x: tf.Tensor, edge_attr: tf.Tensor, edge_paths) -> tf.Tensor:

        num_nodes = tf.shape(x)[0]
        cij = tf.zeros((num_nodes, num_nodes), dtype=tf.float32)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]  
                weight_inds = tf.range(len(path_ij), dtype=tf.int32)
                selected_edge_vector = tf.gather(self.edge_vector, weight_inds)
                selected_edge_attr = tf.gather(edge_attr, path_ij)
                cij = tf.tensor_scatter_nd_update(
                    cij, 
                    indices=[[src, dst]], 
                    updates=[tf.reduce_mean(self.dot_product(selected_edge_vector, selected_edge_attr))]
                )

        cij = tf.where(tf.math.is_nan(cij), tf.zeros_like(cij), cij) 

        return cij

    def dot_product(self, x1, x2) -> tf.Tensor:
        return tf.reduce_sum(x1 * x2, axis=1) 


class MultiHeadAttention(keras.Model):
    def  __init__(self, model_size, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads

        self.WQ = keras.layers.Dense(model_size, name="dense_query")
        self.WK = keras.layers.Dense(model_size, name="dense_key")
        self.WV = keras.layers.Dense(model_size, name="dense_value")

        self.dense = keras.layers.Dense(model_size)

    def call(self, query, key, value, mask):

        batch_size = tf.shape(query)[0]
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])
        
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            score += (1 - mask) * -1e9
        alpha = tf.nn.softmax(score)

        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.model_size))
        output = self.dense(context)
        return output

class GraphHeadAttention(keras.Model):
    def  __init__(self, model_size, num_heads, maxlen,**kwargs):
        super(GraphHeadAttention, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_heads = num_heads
        self.head_size = model_size // num_heads
        self.maxlen = maxlen
        self.edge_encoding = EdgeEncoding(128, num_heads)
        self.spatial_encoding = SpatialEncoding(num_heads)

        self.WQ = keras.layers.Dense(model_size, name="dense_query")
        self.WK = keras.layers.Dense(model_size, name="dense_key")
        self.WV = keras.layers.Dense(model_size, name="dense_value")

        self.dense = keras.layers.Dense(model_size)
        self.b_scale = tf.Variable(initial_value=1e-3, trainable=True, dtype=tf.float32, name="b_scale")
        self.c_scale = tf.Variable(initial_value=1e-3, trainable=True, dtype=tf.float32, name="c_scale")

    def call(self, query, key, value, mask, edge_attr, edge_paths):

        batch_size = tf.shape(query)[0]
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)
        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)

        b = self.spatial_encoding(query, paths=edge_paths)
        b = tf.reduce_mean(b, axis=1, keepdims=True)
        b = tf.reshape(b, (32, 1, 1, 1))        
        b = tf.tile(b, [1, self.num_heads, self.maxlen, self.maxlen]) 
        b = b * self.b_scale

        c = self.edge_encoding(query, edge_attr=edge_attr, edge_paths=edge_paths)
        c = tf.reduce_mean(c, axis=1, keepdims=True)
        c = tf.reshape(c, (32, 1, 1, 1))        
        c = tf.tile(c, [1, self.num_heads, self.maxlen, self.maxlen])  
        c = c * self.c_scale

        score = score + b + c
        if mask is not None:
            score += (1 - mask) * -1e9
        alpha = tf.nn.softmax(score)
        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.model_size))
        output = self.dense(context)
        return output
    
class MyAttn(layers.Layer):
    def __init__(self, model_size, act_ratio=0.25, act_fn=tf.nn.gelu, gate_fn=tf.nn.sigmoid,**kwargs):
        super(MyAttn, self).__init__(**kwargs)
        reduce_channels = int(model_size * act_ratio)
        self.norm = layers.LayerNormalization()
        self.global_reduce = layers.Dense(reduce_channels)
        self.local_reduce = layers.Dense(reduce_channels)
        self.act_fn = act_fn
        self.channel_select = layers.Dense(model_size)
        self.spatial_select = layers.Dense(1)
        self.gate_fn = gate_fn

    def call(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = tf.reduce_mean(x, axis=1, keepdims=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  
        s_attn = self.spatial_select(tf.concat([x_local, tf.tile(x_global, [1, x.shape[1], 1])], axis=-1))
        s_attn = self.gate_fn(s_attn)  

        attn = c_attn * s_attn 
        return ori_x * attn

class MyFNN(keras.Model):
    def __init__(self, dff_size, model_size, drop=0.1,**kwargs):
        super(MyFNN, self).__init__(**kwargs)

        self.fc1 = layers.Dense(dff_size)
        self.act = tf.nn.gelu
        self.fc2 = layers.Dense(dff_size)
        self.attn = MyAttn(dff_size)
        drop = 0.1
        self.drop = layers.Dropout(drop) if drop > 0 else layers.Activation('linear')
        self.fc3 = layers.Dense(model_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x
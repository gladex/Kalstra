import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from LSKAN import *
from GMHA import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

class EncoderLayer(keras.layers.Layer):
    def __init__(self,model_size,num_heads,dff_size,maxlen,rate=0.1,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.attention = GraphHeadAttention(model_size,num_heads,maxlen)
        self.ffn = MyFNN(dff_size,model_size)
        self.resweight1 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.resweight2 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
    
    def call(self,x,FFT_out,mask,edge_attr,edge_paths,training=True):

        attn_output = self.attention(x,FFT_out,FFT_out,mask=mask,edge_attr=edge_attr,edge_paths=edge_paths)
        attn_output = self.dropout1(attn_output,training=training)

        out1 = x + tf.multiply(attn_output,self.resweight1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output,training=training)

        out2 = out1 + tf.multiply(ffn_output,self.resweight2)
        return out2


class Encoder(keras.Model):
    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,rate=0.1,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size,model_size)
        self.pos_embedding = positional_embedding(maxlen,model_size)
        self.encoder_layers = [EncoderLayer(model_size,num_heads,dff_size,maxlen,rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)
    
    def call(self,FFT_out,x,padding_mask,edge_attr,edge_paths,training=True):
        x = self.embedding(x)+self.pos_embedding
        x = self.dropout(x,training=training)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x=x,FFT_out=FFT_out,mask=padding_mask,edge_attr=edge_attr,edge_paths=edge_paths,training=True)
        return x


class DecoderLayer(keras.layers.Layer):
    def __init__(self,model_size,num_heads,dff_size,rate=0.1,**kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.mask_attention = MultiHeadAttention(model_size,num_heads)
        self.attention = MultiHeadAttention(model_size,num_heads)
        self.ffn = MyFNN(dff_size,model_size)

        self.resweight1 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.resweight2 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)
        self.resweight3 = tf.Variable(initial_value=tf.constant([0.000001],dtype=tf.float32), trainable=True)    

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)
    
    def call(self,x,enc_output,look_ahead_mask,padding_mask,training=True):

        attn_decoder = self.mask_attention(x,x,x,look_ahead_mask)
        attn_decoder = self.dropout1(attn_decoder,training=training)

        out1 = x + tf.multiply(attn_decoder,self.resweight1)
        attn_encoder_decoder = self.attention(out1,enc_output,enc_output,padding_mask)
        attn_encoder_decoder = self.dropout2(attn_encoder_decoder,training=training)

        out2 = out1 + tf.multiply(attn_encoder_decoder,self.resweight2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output,training=training)

        out3 = out2 + tf.multiply(ffn_output,self.resweight3)    
        return out3
    
class Decoder(keras.Model):
    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,rate=0.1,**kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.model_size = model_size
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size,model_size)
        self.pos_embedding = positional_embedding(maxlen,model_size)
        self.decoder_layers = [DecoderLayer(model_size,num_heads,dff_size,rate) for _ in range(num_layers)]
        self.droput = keras.layers.Dropout(rate)

    def call(self,enc_output,x,look_ahead_mask,padding_mask,training=True):
        x = self.embedding(x)+self.pos_embedding
        x = self.droput(x,training=training)
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x=x,enc_output=enc_output,look_ahead_mask=look_ahead_mask,padding_mask=padding_mask,training=True)
        return x

def padding_mask(seq):
    mask = tf.cast(tf.math.not_equal(seq,0),dtype=tf.float32)
    mask = mask[:,tf.newaxis,tf.newaxis,:]
    return mask

def look_ahead_mask(size):
    ahead_mask = tf.linalg.band_part(tf.ones((size,size)),-1,0)
    ahead_mask = tf.cast(ahead_mask,dtype=tf.float32)
    return ahead_mask

def create_mask(inp,tar):
    enc_padding_mask = padding_mask(inp)
    dec_padding_mask = padding_mask(tar)
    ahead_mask = look_ahead_mask(tf.shape(tar)[1])
    combined_mask = tf.minimum(dec_padding_mask,ahead_mask)
    return enc_padding_mask,dec_padding_mask,combined_mask


class Transformer(keras.Model):
    def __init__(self,num_layers,model_size,num_heads,dff_size,vocab_size,maxlen,**kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.LSTMencoder = LSTMEncoder(num_layers,model_size,num_heads,dff_size,vocab_size,maxlen)
        self.encoder = Encoder(num_layers,model_size,num_heads,dff_size,vocab_size,maxlen)
        self.decoder = Decoder(num_layers,model_size,num_heads,dff_size,vocab_size,maxlen)
        self.final_dense = tf.keras.layers.Dense(vocab_size,name='final_output')
    
    def call(self,all_inputs,training=True):
        sources1 = all_inputs
        sources2 = all_inputs
        targets = all_inputs
        edge_attr = tf.random.normal((128, 128))
        edge_paths = {i: {j: [0, 1, 2] for j in range(10)} for i in range(10)}
        enc_padding_mask,dec_padding_mask,combined_mask = create_mask(sources1,targets)
        FFT_out = self.LSTMencoder(sources1,padding_mask=enc_padding_mask,training=training)
        enc_output = self.encoder(FFT_out=FFT_out,x=sources2,padding_mask=enc_padding_mask,
                                  edge_attr=edge_attr,edge_paths=edge_paths,training=training)
        dec_output = self.decoder(enc_output=enc_output,x=targets,look_ahead_mask=combined_mask,padding_mask=dec_padding_mask,training=training)
        final_output = self.final_dense(dec_output)
        return final_output
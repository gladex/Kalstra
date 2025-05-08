import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from model import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation, SpatialDropout1D, Convolution1D, GlobalMaxPooling1D
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

class MyDropoutConvPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.2, filters=64, kernel_size=15, **kwargs):
        super(MyDropoutConvPoolingLayer, self).__init__(**kwargs)
        self.dropout = SpatialDropout1D(dropout_rate)
        self.conv = Convolution1D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            kernel_initializer='glorot_normal',
            kernel_regularizer=l2(0.001)
        )
        self.activation = Activation('relu')
        self.global_max_pooling = GlobalMaxPooling1D()

    def call(self, inputs, training=False):
        x = self.dropout(inputs, training=training)
        x = self.conv(x)
        x = self.activation(x)
        return self.global_max_pooling(x)

class PrintEpochMetrics(Callback):
    def __init__(self):
        super(PrintEpochMetrics, self).__init__()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch: {epoch + 1}, Loss: {logs.get('loss'):.4f}, Accuracy: {logs.get('accuracy'):.4f}")

feature = []  
expression = pd.read_csv('finaldata/hlung/train_data.csv')
file = open('finaldata/hlung/train_data.csv')
lines = file.readlines() 
line_0 = lines[0].strip('\n').split(',') 
for i in range(1,len(line_0)):
    tem = list(expression[line_0[i]])    
    feature.append(list(tem))
    file.close()
feature_train = list(feature)

label = []
file = open('finaldata/hlung/train_labels.csv')
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
for i in range(1,len(lable_line_0)):
    label.append(int(lable_line_0[i]))
y_train=[]
for i in label:
    tem =[]
    for j in range(0,15): ###################类别数加1
        tem.append(0)
    tem[i-1]=1
    y_train.append(tem)

feature = []  
testexpression = pd.read_csv('finaldata/hlung/test_data.csv')
file = open('finaldata/hlung/test_data.csv') 
lines = file.readlines() 
line_0 = lines[0].strip('\n').split(',') 
for i in range(1,len(line_0)):
    tem = list(testexpression[line_0[i]])
    feature.append(tem)
file.close()
feature_test = list(feature)

activation = 'relu'
dropout = 0.2
epoch = 100
params_dict = {'kernel_initializer': 'glorot_uniform','kernel_regularizer': l2(0.01),}
num_layers = 4 
model_size = 32
num_heads = 4
dff_size = 128
maxlen = 14###################类别数
vocab_size = 121 

enc_inputs = keras.layers.Input(shape=(maxlen,))
transformer = Transformer(num_layers=num_layers, model_size=model_size, num_heads=num_heads, dff_size=dff_size,
                          vocab_size=vocab_size+1, maxlen=maxlen)
final_output = transformer(enc_inputs,training=True)
dropout_conv_pooling_layer = MyDropoutConvPoolingLayer(dropout_rate=0.2)
final_output = dropout_conv_pooling_layer(final_output, training=True)
final_output = Dense(15,'softmax',**params_dict)(final_output) ######################类别数加1

model = Model(inputs=enc_inputs,outputs=final_output)
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

feature_train = np.array(feature_train)
feature_test = np.array(feature_test)
y_train = np.array(y_train)

model.fit(feature_train, y_train, verbose=0, epochs=epoch, batch_size=32, shuffle=True, callbacks=[PrintEpochMetrics()])
a = model.predict(x=feature_test,batch_size=32)
print(a)

with open('modelsave/hlung_epoch200.txt','w',newline='') as f:
    for i in range(len(a)):
        f.write(str(i))
        f.write(',')
        for j in range(len(a[i])):
            f.write(str(a[i][j]))
            f.write(',')
        f.write('\n')
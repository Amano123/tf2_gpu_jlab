import tensorflow as tf

import tensorflow_datasets as tfds
import os
import io
import glob
import numpy as np
import re


tf.data.TextLineDataset


with open("./sudachi_dataset/バイク.txt", mode="r", encoding="utf-8") as f:
    data = f.readlines()
    print(data[0])


FILE_PATH = "./sudachi_dataset/"
files_path = "./sudachi_dataset/"
DATA_FILE_NAMES = ['バイク.txt', '公開企業.txt']

DATA_FILE_ID = ['0', '1']
print(f"path     :{FILE_PATH}")
print(f"file name:{DATA_FILE_NAMES}")
print(f"file id  :{DATA_FILE_ID}")


def preprocess_sentence(w,num=None):
    w = '<start> ' + w + ' <end>'
    if num == None:
        return w
    else:
        return w,num
        
sample_text  = 'コモドール 社 の コンピューター は 当時 フィンランド で 最も 人気 の ある コンピューター  製品 で あっ た'
print(preprocess_sentence(sample_text))


def labeler(example, index):
    return example, tf.cast(index, tf.int64)
labeled_data_sets = []
for i, file_name in enumerate(DATA_FILE_NAMES, 0):
    lines_dataset = tf.data.TextLineDataset(FILE_PATH + file_name).map(preprocess_sentence)
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)


labeled_data_sets


BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 20000


all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)


for text,label in all_labeled_data:
    print(text.numpy().decode("utf-8"))
    print(label)
    break


tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
vocabulary_0 = set()
vocabulary_1 = set()

for text_tensor, label in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)
    if label.numpy() == 0:
        vocabulary_0.update(some_tokens)
    else:
        vocabulary_1.update(some_tokens)
        
vocab_size   = len(vocabulary_set)
vocab_size_0 = len(vocabulary_0)
vocab_size_1 = len(vocabulary_1)

print(f"full:{vocab_size}")
print(f"0:{vocab_size_0}")
print(f"1:{vocab_size_1}")
print(f"0 + 1 = {len(vocabulary_0 | vocabulary_1)}")


encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def dataset_mapper(token,label):
    token = encoder.encode(token.numpy())
    label= np.array([label])
    return token,label

@tf.function
def tf_encode(token,label):
    return tf.py_function(dataset_mapper, [token,label], [tf.int64, tf.int64])

new_dataset = all_labeled_data.map(tf_encode)


for token, label in all_labeled_data:
    print(token.numpy().decode("utf-8"))
    print(dataset_mapper(token,label))
    break


BATCH_SIZE = 1
new_dataset = new_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1],[-1]),drop_remainder=True)
new_dataset = new_dataset.prefetch(tf.data.experimental.AUTOTUNE)


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


X = tf.keras.Input(shape=(None,), batch_size=BATCH_SIZE)
embedded = tf.keras.layers.Embedding(encoder.vocab_size+1, 128)(X)
lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(128,return_sequences=True,return_state=True, dropout=0.4, recurrent_dropout=0.4)
  )(embedded)
state_h = tf.keras.layers.Concatenate()([forward_h, backward_h]) # 重みを結合
context,attention_weights = Attention(128)(lstm,state_h) # ここにAttentionレイヤを挟む
fully_connected = tf.keras.layers.Dense(units=128, activation='relu')(context)
Y = tf.keras.layers.Dense(1, activation='sigmoid',name='final_layer')(fully_connected)

model = tf.keras.Model(inputs=X, outputs=Y)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


model.fit(new_dataset,epochs=2)


model.save_weights('./test_model/checkpoint')


test = tf.keras.Model(inputs=model.input, outputs=[model.output, model.get_layer('attention').output])


vocab_0 = sum([encoder.encode(vocab) for vocab in vocabulary_0],[])
vocab_1 = sum([encoder.encode(vocab) for vocab in vocabulary_1],[])
print(vocab_0)
print(vocab_1)

score_0 = test.predict([vocab_0])
weght_0 = score_0[1][1]

score_1 = test.predict([vocab_1])
weght_1 = score_1[1][1]


token_to_text_0 = []
for i in vocab_0:
    for x,t in enumerate(vocabulary_set,1):
        # print(x,t)
        if x == int(i):
            token_to_text_0.append(t)
token_to_text_1 = []
for i in vocab_1:
    for x,t in enumerate(vocabulary_set,1):
        # print(x,t)
        if x == int(i):
            token_to_text_1.append(t)


import pandas as pd


df_0 = pd.DataFrame([token_to_text_0, np.ravel(weght_0[0])], index=['text', 'weight']).T
df_1 = pd.DataFrame([token_to_text_1, np.ravel(weght_1[0])], index=['text', 'weight']).T


df_0


df_1


df_0.to_csv("./メディアミックス.csv")
df_1.to_csv("./公開企業.csv")

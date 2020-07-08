# get_ipython().getoutput("pip install tensorflow_datasets pandas")


import tensorflow as tf
import tensorflow_datasets as tfds
import os
import io
import glob
import numpy as np
import re
import pandas as pd


def preprocess_sentence(w, num=None):
    w = '<start> ' + w + ' <end>'
    if num == None:
        return w
    else:
        return w, num


sample_text = '【sample text】'
print(preprocess_sentence(sample_text))


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def create_dataset(file_path, file_names):
    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 20000
    labeled_data_sets = []
    for i, file_name in enumerate(file_names, 0):
        lines_dataset = tf.data.TextLineDataset(os.path.join(file_path, file_name)).map(preprocess_sentence)
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    all_labeled_data = all_labeled_data.shuffle(
                                        BUFFER_SIZE, reshuffle_each_iteration=False
                                        )
    return all_labeled_data


def tokenize(dataset, zero_file_name, one_file_name):
    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()   # all_data_vocab
    vocabulary_zero = set()  # 0 label vocab
    vocabulary_one = set() # 1 label vocab
    for text_tensor, label in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
        if label.numpy() == 0:
            vocabulary_zero.update(some_tokens)
        else:
            vocabulary_one.update(some_tokens)

    vocab_size = len(vocabulary_set)
    vocab_size_0 = len(vocabulary_zero)
    vocab_size_1 = len(vocabulary_one)

    print(f"full:{vocab_size}")
    print(f"0:{vocab_size_0}")
    print(f"1:{vocab_size_1}")
    print(f"0 + 1 = {len(vocabulary_zero | vocabulary_one)}")
    
    with open(f"./vocabulary/vocab_all_{zero_file_name}&{one_file_name}.txt",
                            mode="w", encoding="utf-8") as vocab_all:
        vocab_all.write("\n".join(vocabulary_set))
    
    with open(f"./vocabulary/vocab_{zero_file_name}.txt", 
                            mode="w", encoding="utf-8") as vocab_zero:
        vocab_zero.write("\n".join(vocabulary_set))
    
    with open(f"./vocabulary/vocab_{one_file_name}.txt",
                            mode="w", encoding="utf-8") as vocab_one:
        vocab_one.write("\n".join(vocabulary_set))

    return (
        tokenizer,
        vocabulary_set,
        vocabulary_zero,
        vocabulary_one
    )


# DATA_FILE_NAMES = ['バイク.txt', '公開企業.txt']
# all_labeled_data = create_dataset(file_path=root_path, file_names=DATA_FILE_NAMES)
# tokenize(all_labeled_data, 'バイク', '公開企業')


def dataset_mapper(token,label):
    token = encoder.encode(token.numpy())
    label= np.array([label])
    return token,label

@tf.function
def tf_encode(token,label):
    return tf.py_function(dataset_mapper, [token,label], [tf.int64, tf.int64])

# new_dataset = all_labeled_data.map(tf_encode)


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


def my_model():
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
    
    return model


def split_list(l, n):
    """
    リストをサブリストに分割する
    :param l: リスト
    :param n: サブリストの要素数
    :return: 
    """
    for idx in range(0, len(l), n):
        yield l[idx:idx + n]


def attention_weight(encoder,
                     attention_model,
                     v_zero,
                     v_one,
                     file_name_zero,
                     file_name_one):
    vocab_0 = sum([encoder.encode(vocab) for vocab in v_zero], [])
    vocab_1 = sum([encoder.encode(vocab) for vocab in v_one], [])

    score_0 = attention_model.predict([vocab_0])
    weght_0 = score_0[2]

    score_1 = attention_model.predict([vocab_1])
    weght_1 = score_1[2]
    
    token_to_text_0 = []
    for i in vocab_0:
        for x, t in enumerate(vocabulary_set, 1):
            # print(x,t)
            if x == int(i):
                token_to_text_0.append(t)
    token_to_text_1 = []
    for i in vocab_1:
        for x, t in enumerate(vocabulary_set, 1):
            # print(x,t)
            if x == int(i):
                token_to_text_1.append(t)
    df_0 = pd.DataFrame([token_to_text_0, np.ravel(weght_0[0])], index=['text', 'weight']).T
    df_1 = pd.DataFrame([token_to_text_1, np.ravel(weght_1[0])], index=['text', 'weight']).T
    
    df_0.to_csv(f"./{file_name_zero}.csv")
    df_1.to_csv(f"./{file_name_one}.csv")



# ライブラリのy意味込み
import requests

# メッセージ送信、画像送信、スタンプ送信の処理をクラス化
class LINENotifyBot:
    API_URL = 'https://notify-api.line.me/api/notify'
    def __init__(self, access_token):
        self.__headers = {'Authorization': 'Bearer ' + access_token}

    def send(
            self, message,
            image=None, sticker_package_id=None, sticker_id=None,
            ):
        payload = {
            'message': message,
            'stickerPackageId': sticker_package_id,
            'stickerId': sticker_id,
            }
        files = {}
        if image get_ipython().getoutput("= None:")
            files = {'imageFile': open(image, 'rb')}
        r = requests.post(
            LINENotifyBot.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
            )


bot = LINENotifyBot(access_token='luTtDSuF8bgeUBZoPDyA6s44vHlQO2ptwIWGM8dW7Yp')
def LINE(text):
    bot.send(message=text)


root_path = "./sudachi_dataset/"
file_names = [i for i in os.listdir(root_path) if i[-4:] == ".txt"]
pair_datas = list(split_list(file_names, 2))
line_str = ""
for i in pair_datas:
    line_str += i[0] + i[1]

LINE(f"\n{line_str}の学習を始めるンゴ")

LINE("\n機械学習スタート")

for num, pair_data in enumerate(pair_datas, 0):
    LINE(f"\n{pair_data[0]}と{pair_data[1]}の学習を開始します")
    all_labeled_data = create_dataset(file_path=root_path, file_names=pair_data)
    tokenizer, vocabulary_set, vocabulary_zero, vocabulary_one = tokenize(all_labeled_data,
                                                                          pair_data[0][:-4],
                                                                          pair_data[1][:-4])
    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    new_dataset = all_labeled_data.map(tf_encode)
    BATCH_SIZE = 1
    new_dataset = new_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]), drop_remainder=True)
    new_dataset = new_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    model = my_model()
    
    model.fit(new_dataset,epochs=2)
    
    model.save_weights(f'./datalogs/{pair_data[0]}&{pair_data[1]}/checkpoint')
    test = tf.keras.Model(inputs=model.input,
                          outputs=[
                              model.output,
                              model.get_layer(index=-3).output
                          ])
    attention_weight(encoder,
                     test,
                     vocabulary_zero,
                     vocabulary_one,
                     pair_data[0][:-4],
                     pair_data[1][:-4])
    LINE(f"\n{pair_data[0]}と{pair_data[1]}の学習終了")


vocab_0 = sum([encoder.encode(vocab) for vocab in vocabulary_zero], [])

score_0 = test.predict([vocab_0])
weght_0 = score_0[2]


len(np.ravel(weght_0[0]))


len(np.ravel(score_0[0])), len(np.ravel(score_0[1])), len(np.ravel(score_0[2]))


# root_path = "./sudachi_dataset/"
# file_paths = glob.glob("")
# DATA_FILE_NAMES = ['バイク.txt', '公開企業.txt']
# all_labeled_data = create_dataset(file_path=root_path, file_names=DATA_FILE_NAMES)

# tokenizer, vocabulary_set, vocabulary_zero, vocabulary_one = tokenize(all_labeled_data)
# tokenizer, len(vocabulary_set), len(vocabulary_zero), len(vocabulary_one)

# encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

# BATCH_SIZE = 1
# new_dataset = new_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1],[-1]),drop_remainder=True)
# new_dataset = new_dataset.prefetch(tf.data.experimental.AUTOTUNE)
# model = my_model()
# model.fit(new_dataset,epochs=2)


# model = my_model()


# tokenizer, vocabulary_set, vocabulary_zero, vocabulary_one = tokenize(all_labeled_data)
# tokenizer, len(vocabulary_set), len(vocabulary_zero), len(vocabulary_one)


# encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


# BATCH_SIZE = 1
# new_dataset = new_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1],[-1]),drop_remainder=True)
# new_dataset = new_dataset.prefetch(tf.data.experimental.AUTOTUNE)


# model.fit(new_dataset,epochs=2)

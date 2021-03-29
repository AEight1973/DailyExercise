import re, json, os
import time
import graphviz
import gensim
import gridfs
import openpyxl
import multiprocessing
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import xlrd
import xlwt
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import plot_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pymongo
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
import heapq



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# tf.compat.v1.disable_eager_execution()

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

'''生成式文本摘要'''


# myclient = pymongo.MongoClient('mongodb://localhost:27017/')
# mydb = myclient['caixin']
# list_set = ['economy', 'finance', 'companies', 'china', 'international']
# article = 0
# content_set = []
# subhead_set = []
# for i in range(len(list_set)):
#     table = mydb[list_set[i]]
#     lines = table.find()
#
#     if lines is not None:
#         for item in lines:
#             content = item['content']
#             subhead = item['subhead']
#             content = content.lstrip('content:')
#             subhead = subhead.lstrip('subhead:')
#             if content is not None and subhead is not None:
#                 content_set.append(content)
#                 subhead_set.append(subhead)
#                 article += 1
#
# print("article={}".format(article))


def remove_stopwords(words, path):
    with open(path, 'r', encoding="utf-8") as f:  # 读取停用词
        stopwords = f.read().split("\n")
    filtered_words = []
    for word in words:
        if word not in stopwords:
            filtered_words.append(word)
    return filtered_words


def article2word(article):
    # 对句子替换标点
    article_list = []
    a_append = article_list.append
    for i in range(len(article)):
        a_append(
            re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；：:-【】+\"\']+|[+——！，;：:。？、~@#￥%……&*（）]+", "", article[i]))  # 替换标点符号
    # print(article_list)

    # 分词
    # jieba.enable_paddle()
    total_wordlist1, total_wordlist2 = [], []  # 以文章为单位的分词list，元素为str
    t1_append = total_wordlist1.append
    t2_append = total_wordlist2.append
    print("分词进度：")
    for i in tqdm(article_list):
        # 导入用户自定义字典
        # jieba.load_userdict("userdict.txt")
        words = jieba.lcut(i)  # 为了速度先不用paddle
        # words=jieba.lcut(i,use_paddle=True)  #如果paddle分词的str是一个空白符，那么将直接停止运行并报错
        # words_list = remove_stopwords(words, 'stopwords-master\hit_stopwords.txt')  # 去停用词
        t1_append(words)
        t2_append(" ".join(words))
    return total_wordlist1, total_wordlist2


# df = pd.read_excel("summarization.xls")
# newcontent = df["抽取摘要"].values.tolist()
# subhead_set = df["原subhead"].values.tolist()
# print(newcontent[:10])

df = pd.read_csv("./Pre LCSTS/train.csv", header=None)
df_li = df.values.tolist()
newcontent = [i[1] for i in df_li]
subhead_set = [i[0] for i in df_li]
print(newcontent[:10])
print(subhead_set[:10])

# x_data = article_to_word(list(content_set))
# print(x_data[:10])
# arc2word=article2word(content_set)
# print(arc2word[:10])

# 得到分词后用list分开和用空格分开的article
content2word, x_data = article2word(newcontent)
print(x_data[:10])
print(content2word[:10])
subhead2word, _ = article2word(subhead_set)
print(subhead2word[:10])

max_length = max([len(s.split()) for s in x_data])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_data)
vocab = tokenizer.word_index
vocad_size = len(tokenizer.word_index) + 1
print('词汇表大小：', vocad_size)


# 构建词汇表索引
def process_data_index(datas, vocab2id):
    data_indexs = []
    append = data_indexs.append
    for words in datas:
        line_index = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in words]
        append(line_index)
    return data_indexs


vocab_words = list(vocab)
special_words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
vocab_words = special_words + vocab_words
vocab2id = {word: i for i, word in enumerate(vocab_words)}
id2vocab = {i: word for i, word in enumerate(vocab_words)}

source_data = content2word
source_data_ids = process_data_index(source_data, vocab2id)
target_data = subhead2word
target_data_ids = process_data_index(target_data, vocab2id)

print("vocab test: ", [id2vocab[i] for i in range(10)])
print("source test: ", source_data[10])
print("source index: ", source_data_ids[10])
print("target test: ", target_data[10])
print("target index: ", target_data_ids[10])


def process_decoder_input_output(target_indexs, vocab2id):
    decoder_inputs, decoder_outputs = [], []
    i_append, o_append = decoder_inputs.append, decoder_outputs.append
    for target in target_indexs:
        i_append([vocab2id["<GO>"]] + target)
        o_append(target + [vocab2id["<EOS>"]])
    return decoder_inputs, decoder_outputs


target_input_ids, target_output_ids = process_decoder_input_output(target_data_ids, vocab2id)
print("decoder inputs: ", target_input_ids[:2])
print("decoder outputs: ", target_output_ids[:2])

maxlen = max_length
source_input_ids = keras.preprocessing.sequence.pad_sequences(source_data_ids, padding='post', maxlen=maxlen)
target_input_ids = keras.preprocessing.sequence.pad_sequences(target_input_ids, padding='post', maxlen=maxlen)
target_output_ids = keras.preprocessing.sequence.pad_sequences(target_output_ids, padding='post', maxlen=maxlen)
print(source_data_ids[:5])
print(target_input_ids[:5])
print(target_output_ids[:5])

maxlen = max_length
embedding_dim = 50
hidden_units = 128
vocab_size = len(vocab2id)

"""
seq2seq model
"""
# Input Layer
encoder_inputs = Input(shape=(maxlen,), name="encode_input")
decoder_inputs = Input(shape=(None,), name="decode_input")
# Encoder Layer
enc_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="encoder_embedding")
encoder_embed = enc_embedding(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="encode_lstm")
enc_outputs, enc_state_h, enc_state_c = encoder_lstm(encoder_embed)
dec_states_inputs = [enc_state_h, enc_state_c]
# Decoder Layer
dec_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding")
decoder_embed = dec_embedding(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="decode_lstm")
dec_outputs, _, _ = decoder_lstm(decoder_embed, initial_state=dec_states_inputs)
decoder_attention = Attention()
attention_output = decoder_attention([dec_outputs, enc_outputs])
# Dense Layer
decoder_dense = Dense(vocab_size, activation='softmax', name="dense")
dec_outputs = decoder_dense(attention_output)
# seq2seq model
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dec_outputs)
plot_model(model, to_file="summarization.png", show_shapes=True, show_layer_names=False)

epochs = 100
batch_size = 64
val_rate = 0.1

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam', metrics=['acc'])
model.fit([source_input_ids, target_input_ids], target_output_ids,
          batch_size=batch_size, epochs=epochs, validation_split=val_rate, verbose=1)

# save model weights
model.save("data/seq2seq_attention_weights.h5")
del model

K.clear_session()

# load model weights
model = load_model("data/seq2seq_attention_weights.h5")
print(model.summary())

# encoder model
encoder_model = Model(inputs=encoder_inputs, outputs=[enc_outputs] + dec_states_inputs)
print(encoder_model.summary())

# decoder model
enc_output = Input(shape=(maxlen, hidden_units), name='enc_output')
enc_input_state_h = Input(shape=(hidden_units,), name='input_state_h')
enc_input_state_c = Input(shape=(hidden_units,), name='input_state_c')
dec_input_states = [enc_input_state_h, enc_input_state_c]

dec_outputs, out_state_h, out_state_c = decoder_lstm(decoder_embed, initial_state=dec_input_states)
dec_output_states = [out_state_h, out_state_c]
decoder_attention = Attention()
attention_output = decoder_attention([dec_outputs, enc_output])

decoder_dense = Dense(vocab_size, activation='softmax', name="dense")
dense_output = decoder_dense(attention_output)

decoder_model = Model(inputs=[enc_output, decoder_inputs, dec_input_states], outputs=[dense_output] + dec_output_states)
print(decoder_model.summary())


def infer_predict(input_text, encoder_model, decoder_model):
    text_words = input_text.split()[:maxlen]
    input_id = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in text_words]
    input_id = [vocab2id["<GO>"]] + input_id + [vocab2id["<EOS>"]]
    if len(input_id) < maxlen:
        input_id = input_id + [vocab2id["<PAD>"]] * (maxlen - len(input_id))

    input_source = np.array([input_id])
    input_target = np.array([vocab2id["<GO>"]])

    # 编码器encoder预测输出
    enc_outputs, enc_state_h, enc_state_c = encoder_model.predict([input_source])
    dec_inputs = input_target
    dec_states_inputs = [enc_state_h, enc_state_c]

    result_id = []
    result_text = []
    i_append = result_id.append
    t_append = result_text.append
    for i in range(maxlen):
        # 解码器decoder预测输出
        dense_outputs, dec_state_h, dec_state_c = decoder_model.predict([enc_outputs, dec_inputs] + dec_states_inputs)
        pred_id = np.argmax(dense_outputs[0][0])
        i_append(pred_id)
        t_append(id2vocab[pred_id])
        if id2vocab[pred_id] == "<EOS>":
            break
        dec_inputs = np.array([[pred_id]])
        dec_states_inputs = [dec_state_h, dec_state_c]
    return result_id, result_text


def infer_encoder_output(input_text, encoder, maxlen=max_length):
    text_words = input_text.split()[:maxlen]
    input_id = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in text_words]
    input_id = [vocab2id["<GO>"]] + input_id + [vocab2id["<EOS>"]]
    if len(input_id) < maxlen:
        input_id = input_id + [vocab2id["<PAD>"]] * (maxlen - len(input_id))
    input_source = np.array([input_id])
    # 编码器encoder预测输出
    enc_outputs, enc_state_h, enc_state_c = encoder.predict([input_source])
    enc_state_outputs = [enc_state_h, enc_state_c]
    return enc_outputs, enc_state_outputs


def infer_beam_search(enc_outputs, enc_state_outputs, decoder, k=5):
    dec_inputs = [vocab2id["<GO>"]]
    states_curr = {0: enc_state_outputs}
    seq_scores = [[dec_inputs, 1.0, 0]]

    for _ in range(maxlen):
        cands = list()
        states_prev = states_curr
        for i in range(len(seq_scores)):
            seq, score, state_id = seq_scores[i]
            dec_inputs = np.array(seq[-1:])
            dec_states_inputs = states_prev[state_id]
            # 解码器decoder预测输出
            dense_outputs, dec_state_h, dec_state_c = decoder.predict([enc_outputs, dec_inputs] + dec_states_inputs)
            prob = dense_outputs[0][0]
            states_curr[i] = [dec_state_h, dec_state_c]

            for j in range(len(prob)):
                cand = [seq + [j], score * prob[j], i]
                cands.append(cand)

        seq_scores = heapq.nlargest(k, cands, lambda d: d[1])

    res = " ".join([id2vocab[i] for i in seq_scores[0][0]])
    return res


newdf = pd.read_csv("./Pre LCSTS/test.csv", header=None)
newdf_li = newdf.values.tolist()
test_content = [i[1] for i in newdf_li]
_, input = article2word(test_content)
# for input_text in tqdm(input):
#     result_id, result_text = infer_predict(input_text, encoder_model, decoder_model)
#     print("Input: ", input_text)
#     print("Output: ", result_text, result_id)


for input_text in tqdm(input):
    enc_outputs, enc_state_outputs = infer_encoder_output(input_text, encoder_model, max_length)
    res = infer_beam_search(enc_outputs, enc_state_outputs, decoder_model)
    print("Input: ", input_text)
    print("Output: ", res)

# -*- coding=utf-8 -*-
import re
import math
import time
import string
import cPython
import graphviz
import gensim
import keras
import openpyxl
import multiprocessing
from collections import Counter
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import xlrd
import xlwt
import tensorflow as tf
from sklearn import metrics,svm
from sklearn.manifold import TSNE
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Flatten, Embedding, Dropout, BatchNormalization, concatenate, Input, MaxPool1D, Layer, \
    InputSpec
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from gensim.models import FastText
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import nltk
import os



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

data=pd.read_excel('last-data.xlsx',engine="openpyxl")
data=data.values.tolist()[0:4439]

old_x_data,y_data=[],[]
for i in range(len(data)-1):
    if data[i][6] == "Positive ID":
        data[i][6]=0
    if data[i][6] == "Negative ID":
        data[i][6]=1
    if data[i][6] == "Unverified":
        data[i][6]=2
    if data[i][2] != ' ':
        old_x_data.append(str(data[i][2]))
        y_data.append(data[i][6])

x_data=[i.lower() for i in old_x_data]
#print(x_data[:5])

#划分训练集、测试集，比例7:3
def data_split(data,label,ratio_test):
    x_train, x_test, y_train, y_test = train_test_split(data,label,test_size=ratio_test )
    return x_train, x_test, y_train, y_test

x_train,x_test, y_train, y_test=data_split(x_data,y_data,0.25)
#print(x_train[:5],y_train[0:5])

#文本处理
def remove_stopwords(words,path):
    with open(path, 'r', encoding="utf-8") as f:  # 读取停用词
        stopwords = f.read().split("\n")
    filtered_words=[]
    for word in words:
        if word not in stopwords:
            filtered_words.append(word)
    return filtered_words

#以文章为单位的分词
def article_to_word(article):
    #对句子替换标点
    article_list=[]
    for i in range(len(article)):
        article_list.append(re.sub("[^A-Za-z0-9]", " ", article[i])) # 替换标点符

    #分词
    total_wordlist=[]  #以文章为单位的分词list，元素为str
    print("分词进度：")
    for i in tqdm(article_list):
        words = nltk.word_tokenize(i)
        words_list = remove_stopwords(words,'stopword.txt')  #去停用词
        total_wordlist.append(" ".join(words_list))
    return total_wordlist


x_data=article_to_word(x_data)
x_train=article_to_word(x_train)
x_test=article_to_word(x_test)
print(x_data[:10])


max_length=max([len(s.split()) for s in x_data])
print("最长文本大小:",max_length)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(x_data)
vocab=tokenizer.word_index
vocad_size=len(tokenizer.word_index)+1
print('词汇表大小：',vocad_size)

def encode_docs(tokenizer,max_length,docs):
    encoded=tokenizer.texts_to_sequences(docs)     #单词-整数映射
    padded=pad_sequences(encoded,maxlen=max_length)
    return padded

x_train=encode_docs(tokenizer,max_length,x_train)
x_test=encode_docs(tokenizer,max_length,x_test)
#print(np.shape(x_train), np.shape(x_test))
y_train=np.array(y_train)
y_test=np.array(y_test)
y_train=keras.utils.to_categorical(y_train, num_classes=3)   #3分类
y_test_val=keras.utils.to_categorical(y_test, num_classes=3)


#模型建好后可以直接使用
t1 = time.time()
textmodel=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
t2 = time.time()
print("成功加载model，加载时间为：{:.4f}".format(t2-t1))

#textmodel=FastText(size=300, window=3, min_count=1)a

# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((vocad_size, 300))
for word, i in vocab.items():
    try:
        embedding_vector = textmodel.wv[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue

print("词向量矩阵构建成功")

# 构建TextCNN模型
#模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接
def textcnn(max_sequence_length, max_token_num, embedding_dim, output_dim,
            embedding_matrix, model_img_path=None):
    main_input = Input(shape=(max_sequence_length,),dtype='float64')

    embed = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                          weights=[embedding_matrix], trainable=False)(main_input)
    pool_output = []
    kernel_sizes = [2,3,4]
    for kernel_size in kernel_sizes:
        cnn = Conv1D(filters=512, kernel_size=kernel_size, strides=1, activation="relu")(embed)
        pool = MaxPool1D(pool_size=int(cnn.shape[1]))(cnn)
        pool_output.append(pool)
    pool_output = concatenate([p for p in pool_output])
    flat = Flatten()(pool_output)
    drop = Dropout(0.5)(flat)
    main_output = Dense(output_dim, activation='softmax')(drop)
    model = Model(inputs=main_input,outputs=main_output)
    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model=textcnn(max_sequence_length=max_length,max_token_num=vocad_size,embedding_dim=300,output_dim=3,
               embedding_matrix=embedding_matrix,model_img_path="model.png")


plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=False)
# model.fit(x_train, y_train, batch_size=50, epochs=100, verbose=1)

earlystop = EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=1)
model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1,
          validation_data=(x_test, y_test_val),callbacks=[earlystop])
mc = ModelCheckpoint(filepath='best_model.h5',monitor='val_acc',mode='max',verbose=1,save_best_only=True)
model.save('word2vec_classify.h5')


#分类器评估
model=load_model('word2vec_classify.h5')
#y_test_onehot = keras.utils.to_categorical(y_test, num_classes=5)  # 将标签转换为one-hot编码
predict_y=model.predict(x_test)
y_pred=np.argmax(predict_y,axis=1)
print(classification_report(y_test.astype(int),y_pred.astype(int)))
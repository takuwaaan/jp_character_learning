#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

#認識用
import scipy.misc
from keras import backend as K
from keras import initializers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from PIL import Image

#データ呼び出し
ary = np.load("ETL7.npz")['arr_0'].reshape([-1, 32, 32]).astype(np.float32) / 15

#データ内容の確認
plt.figure()
plt.imshow(ary[3900])
plt.colorbar()
plt.grid(False)
plt.show()

#認識工程
#訓練データとテストデータへ変換

#ひらがな46文字
nb_classes = 46
#画像サイズ(32,32)
img_rows, img_cols = 32, 32

#350人のひらがなデータを訓練データとして用意
#scipy.cisc.imresizeは古いので最新の方法に変更 https://walkingmask.hatenablog.com/entry/2019/08/09/205651
X_train = np.zeros([nb_classes * 350, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 350):
    X_train[i] = np.array(Image.fromarray(ary[i]).resize((img_rows,img_cols),resample=2))
    # X_train[i] = ary[i]  
Y_train = np.repeat(np.arange(nb_classes), 350)

#トレーニングデータとテストデータに分割
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

# 画像集合を表す4次元テンソルに変形
# keras.jsonのimage_dim_orderingがthのときはチャネルが2次元目、tfのときはチャネルが4次元目にくる
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#ラベル情報を２進数へ変換（Kerasで必要）
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

#訓練データへのノイズ関数適用（１５度回転、1.2or0.8ズーム）
datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

#CNN
model = Sequential()

def m6_1():
    model.add(Convolution2D(32,  (3, 3),input_shape=input_shape))
    model.add(Convolution2D(32,  (3, 3),input_shape=input_shape))
    model.add(Convolution2D(32,  (3, 3),input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

#認識実行(epoch:10)
m6_1()
# classic_neural()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16), samples_per_epoch=X_train.shape[0],
                    nb_epoch=10, validation_data=(X_test, Y_test))

#正解率の表示
score = model.evaluate(X_test, Y_test, verbose=1)
print("正解率(acc)：", score[1])

#予測モデルの保存
model.save("Hiragana_97_e10.h5")





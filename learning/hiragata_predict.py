#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

#OpenCVでの描画

#sx, syは線の始まりの位置
sx, sy = 0, 0
#ペンの色
color = (0, 0, 0)
#ペンの太さ
thickness = 10

#筆跡記憶配列
dtype_plot = [("x", "i2"), ("y", "i2")]
plot = np.zeros(1024, dtype=dtype_plot)
plot_size = 0

#筆跡記憶配列　３次元配列ver
#np.full(((線の本数、描画数、x・y)))
data = np.full(((100,1000,2)),-100)
n = 0

#マウスの操作があるとき呼ばれる関数
def callback(event, x, y, flags, param):
    global img, sx, sy, color, thickness, plot_size,data,n

    #(STATUS)
    #event
    #CV_EVENT_MOUSEMOVE (0):マウスカーソルが動いた
    #CV_EVENT_LBUTTONDOWN (1):左ボタンが押された
    #CV_EVENT_RBUTTONDOWN (2):右ボタンが押された
    #CV_EVENT_MBUTTONDOWN (3):中央ボタンが押された
    #CV_EVENT_LBUTTONUP (4):左ボタンが離された
    #CV_EVENT_RBUTTONUP (5):右ボタンが離された
    #CV_EVENT_MBUTTONUP (6):中央ボタンが離された
    #CV_EVENT_LBUTTONDBLCLK (7):左ボタンがダブルクリックされた
    #CV_EVENT_RBUTTONDBLCLK (8):右ボタンがダブルクリックされた
    #CV_EVENT_MBUTTONDBLCLK (9):中央ボタンがダブルクリックされた
    #flags
    #CV_EVENT_FLAG_LBUTTON (1):マウス左ボタンが押されているか
    #CV_EVENT_FLAG_RBUTTON (2):マウス右ボタンが押されているか
    #CV_EVENT_FLAG_MBUTTON (4):マウス中央ボタンが押されているか
    #CV_EVENT_FLAG_CTRLKEY (8):CTRLボタンが押されているか
    #CV_EVENT_FLAG_SHIFTKEY (16):Shiftボタンが押されているか
    #CV_EVENT_FLAG_ALTKEY (32):Altボタンが押されているか
    
    #マウスの左ボタンがクリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN:
        sx, sy = x, y
        data[n,plot_size] = (x*100/512,y*100/512)
        plot_size +=1       

    #マウスの左ボタンがクリックされていて、マウスが動いたとき
    if (flags & cv2.EVENT_FLAG_LBUTTON) and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, (sx, sy), (x, y), color, thickness)
        sx, sy = x, y
        data[n,plot_size] = (x*100/512,y*100/512)
        plot_size += 1
    
    #マウスの左ボタンがクリックされていて、マウスが離れたとき
    if (flags & cv2.EVENT_FLAG_LBUTTON) and event == cv2.EVENT_LBUTTONUP:
        n+=1
        plot_size = 0
        
#新しいウィンドウを開く
img = np.zeros((512, 512, 1), np.uint8)
img[:]=(255)
cv2.namedWindow("img")

#マウス操作のコールバック関数の設定
cv2.setMouseCallback("img", callback)

while(1):
    cv2.imshow("img", img)
    k = cv2.waitKey(1)

    #Escキーを押すと終了
    if k == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break

    #sを押すとデータを保存
    if k == ord("s"):
        path_w= input("filename=>")
        cv2.imwrite(path_w+".jpg",img)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break
        

    #cを押すとデータクリア
    if k == ord("c"):
        img[:]=(255)
        data = np.full(((100,1000,2)),-100)
        plot_size=0
        n=0
        print ("clear")
        
#予測用へ保存したデータを変換

Xt = []
img = cv2.imread(path_w+".jpg", 0)
img = cv2.resize(img,(32, 32), cv2.INTER_CUBIC)
img = cv2.bitwise_not(img) # 白黒反転
img = cv2.GaussianBlur(img,(9,9), 0)

#データ内容（手書き文字）の確認
plt.figure()
plt.imshow(img)
plt.colorbar()
plt.grid(False)
plt.show()

Xt.append(img)
Xt = np.array(Xt)

#学習時のデータ形式と合わせる

img_rows, img_cols = 32, 32
# 画像集合を表す4次元テンソルに変形
# keras.jsonのimage_dim_orderingがthのときはチャネルが2次元目、tfのときはチャネルが4次元目にくる
if K.image_dim_ordering() == 'th':
    Xt = Xt.reshape(Xt.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    Xt = Xt.reshape(Xt.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#予測モデルの呼び込み
model = load_model("Hiragana_99_e40.h5")

#予測結果出力
result = model.predict_classes(Xt)
model.predict_classes
h = result[0]
hiragana = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"]
print("この文字は : 「"+hiragana[h]+"」です")


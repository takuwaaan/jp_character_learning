#!/usr/bin/env python
# coding: utf-8

# In[42]:


#ETL7img2data.py 
#画像を読み込み，ひらがな画像データセットを作成
# 使用するライブラリを読み込む
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pickle

# 保存ディレクトリと画像サイズの指定
out_dir = "./ETL7-img" # ひらがな画像集のディレクトリ
im_size = 32 # 画像サイズ

save_file = out_dir + "/hiragana.pickle" # 保存ファイル名と保存先
plt.figure(figsize=(9, 17)) # 出力画像を大きくする

# ひらがな画像集のディレクトリから画像を読み込み開始
hiraganadir = list(range(177, 220+1)) #あいうえお--わの順序
hiraganadir.append(166) # を
hiraganadir.append(221) # ん
result = []
#[types,data,size,size]
ary = np.zeros([46, 350, 32, 32], dtype=np.uint8)

for i, code in enumerate(hiraganadir):
   img_dir = out_dir + "/" + str(code)
   fs = glob.glob(img_dir + "/*")
   print("dir=",  img_dir)

   # 画像64X63を読み込んでグレイスケールに変換し32X32に整形
   for j, f in enumerate(fs):
       img = cv2.imread(f)
       img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       img = cv2.resize(img_gray, (im_size, im_size))
       result.append([i, img])
       #ndarrayで保存
       ary[i, j] = img

       # ひらがな画像一覧表示 10行X5列
       if j == 2:
           plt.subplot(11, 5, i + 1)
           plt.axis("off")
           plt.title(str(i))
           plt.imshow(img, cmap='gray')


# In[43]:


ary.shape


# In[44]:


img


# In[45]:


ary[0]


# In[46]:


np.savez_compressed("ETL7.npz", ary)


# In[ ]:





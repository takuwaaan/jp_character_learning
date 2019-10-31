#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ETL7toimg.py ETL7の4個の分割ファイルを読み，ひらがな画像集を作成
# 使用するライブラリを読み込む
import struct
from PIL import Image, ImageEnhance
import glob, os

# ひらがな画像集を保存するディレクトリ
outdir = "ETL7-img/"
if not os.path.exists(outdir): os.mkdir(outdir)

# ETL7ディレクトリ内部の4個の分割データを読み込む
files = glob.glob("ETL7/*")
for fname in files:
   if fname == "ETL7/ETL7INFO": continue # 不要な分割ファイルは省く
   print(fname) # ETL7の分割ファイル名

   # ETL7の分割ファイル名を開く
   f = open(fname, 'rb')
   f.seek(0)
   while True:
       # あいうえおのラベルと画像データの組を一つずつ読む
       s = f.read(2052)
       if not s: break
       # バイナリデータなのでPythonが理解できるように
       r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
       code_ascii = r[1]
       code_jis = r[3]
       # ひらがなの画像として取り出す
       iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
       iP = iF.convert('L')
       dir = outdir + "/" + str(code_jis)
       if not os.path.exists(dir): os.mkdir(dir)
       fn = "{0:02x}-{1:02x}{2:04x}.png".format(code_jis, r[0], r[2])
       fullpath = dir + "/" + fn
       #if os.path.exists(fullpath): continue
       enhancer = ImageEnhance.Brightness(iP)
       iE = enhancer.enhance(16)
       iE.save(fullpath, 'PNG')
print("ok")


# In[ ]:





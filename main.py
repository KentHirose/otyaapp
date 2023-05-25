import streamlit as st
from PIL import Image
import numpy as np

st.title("🍵 緑茶のパッケージから味を予測")

# init
# 画像のパス
picture = None
# 画像のnp配列
pic_array = None
# 味のnp配列
taste = None
# 甘味、渋味、苦味、香りの配列
taste_column = {"甘味": None, "渋味": None, "苦味": None, "香り": None}


agree = st.checkbox('📷 Start taking pictures')


# カメラの許可
if agree:
    # 写真の取得
    picture = st.camera_input("Take a picture")

# 写真の配列化
if picture is not None:
    
    # PILで開く
    pil_pic = Image.open(picture)
    
    # 配列化
    pic_array = np.array(pil_pic)
    st.write(pic_array.shape)
    st.write(pic_array)

st.markdown('### 📈 Taste Prediction')

# 配列を味に変換
if pic_array is not None:
    # [0.5, 0.5, 0.5, 0.5]のnp.arrayと仮定する
    taste = np.array([0.5, 0.5, 0.5, 0.5])
    
    # taste_columnに代入
    for i in range(4):
        taste_column[list(taste_column.keys())[i]] = taste[i]
    
    st.markdown('##### 👀 このお茶の味は...')
    
    # 項目ごとに表示
    st.write(taste_column)

# 綾鷹の例
st.markdown('##### 参考：綾鷹の味')
st.write({"甘味": 0.4, "渋味": 0.7, "苦味": 0.2, "香り": 0.1})

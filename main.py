import streamlit as st
from PIL import Image
import numpy as np
import cv2
import numpy as np
import torch
import torchvision

# 関数
# 画像をRGBに変換する関数
def img_to_rgb(img):
    Img = cv2.imread(img)
    # Faster R-CNNの事前学習済みモデルをロード
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 推論モードに設定する
    model.eval()

    # 画像をテンソルに変換する
    tensor_Image = torchvision.transforms.functional.to_tensor(Img)

    # モデルに画像を入力して物体検出を実行する
    output = model([tensor_Image])

    # 物体検出結果を取得する
    Boxes = output[0]['boxes'].detach().numpy()  
    Labels = output[0]['labels'].detach().numpy()  
    Scores = output[0]['scores'].detach().numpy()  

    # 最も信頼度が高い物体のインデックス，バウンディングボックス，クラスラベルを取得する
    maxtrust_index = np.argmax(Scores)
    maxtrust_box = Boxes[maxtrust_index]
    maxtrust_label = Labels[maxtrust_index]

    # 最も信頼度が高い物体のRGB値を算出する
    x1, y1, x2, y2 = maxtrust_box.astype(int)
    roi = Img[y1:y2, x1:x2]  # バウンディングボックスで切り出した領域
    r = int(np.mean(roi[:,:,0]))
    g = int(np.mean(roi[:,:,1]))
    b = int(np.mean(roi[:,:,2]))
    rgb = np.array([r, g, b]).astype(int)
    return rgb

# RGBを特徴色に変える関数
def comp_sim(rgb):
    sim = []
    gyb = np.array([
        [0, 255, 0], 
        [255, 255, 0], 
        [255, 255, 255]
    ])
    for i in gyb:
        sim.append(np.dot(rgb, i) / (np.linalg.norm(rgb) * np.linalg.norm(i)))
    return np.array(sim)

# 表現行列を作成する関数
def make_hyougen_matrix(tastes, pack_colors):
    """
    相関係数を利用して表現行列を求める
    
    Parameters
    -----------
    tastes : array(2次元5行4列)
        お茶の味ベクトル
        甘味、渋味、苦味、香りの順に格納されたベクトルがお茶の数分入っている行列
        [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
    pack_colors : array(2次元5行3列)
        お茶のパッケージカラーのベクトル
        緑、黄色、黒の順に格納されたベクトルがお茶の数分入っている行列
        [[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]

    Returns
    ------
    feature_matrix : 表現行列

    """
    # 総当たりで，各要素の相関係数を求める
    ## 各要素のみを集めたベクトルを作成する
    ### パッケージの色
    ### color = [ green, yellow, black ]

    
    green = np.array([ color[0] for color in pack_colors])
    yellow =np.array( [ color[1] for color in pack_colors])
    black = np.array( [ color[2] for color in pack_colors])

    ### 味
    ### taste = [sweet, astrin, bitter, scent]
    sweet = np.array([ taste[0] for taste in tastes])
    astrin =np.array( [ taste[1] for taste in tastes])
    bitter =np.array( [ taste[2] for taste in tastes])
    scent = np.array([ taste[3] for taste in tastes])

    ## 相関係数を求める
    ## 以下で何をしているかの説明は以下のリンク参照してください
    pack_color_matrix = [green, yellow, black]
    taste_matrix = [sweet, astrin, bitter, scent]
    pre_hyougen_matrix = np.corrcoef(pack_color_matrix, taste_matrix)
    ### 求めた相関係数から、必要なところだけを取り出している
    hyougen_matrix = pre_hyougen_matrix[0:3,3:7]
    return hyougen_matrix
    ## エクセル
    ## https://docs.google.com/spreadsheets/d/1my61VevGl9skGzP4Do9pXqiUOJHKfsoQtOU2vY_r7y8/edit?usp=sharing
    ##スライド
    ## https://docs.google.com/presentation/d/1bEPOm9R-RysHFgoKSL2qh1MxqRaw20-aEyXLns0adfk/edit#slide=id.g24b6131f0d1_0_15
    
# 表現行列を元に味を予測する関数
def conversion_rgb_into_taste(A,x):
    """
    RGBから味を求める
    これで積は求められるけど，計算の順番が逆ではある

    Parameter
    --------------------
    A : array(2次元3行4列)
        表現行列(下に書かれてるやつは例です)
        [[-0.41267342 -0.43412746  0.69778051  0.99826524]
        [-0.18818657 -0.59925896  0.56089673  0.79344375]
        [ 0.39462315 -0.34373807 -0.16478485 -0.29246541]]
    x : array(1次元1行3列)
        入力された画像のRGBを緑、黃、黒に変換したベクトル

    Returns
    --------------------
    taste_score : array(1次元1行4列)
        それぞれの確率
        
    """
    # y = Axの部分の計算
    taste_score = np.dot(x,A)
    return taste_score

# 表現行列の作成過程
# サンプル画像に対応する特徴色のデータを作成した
# サンプル画像に対応する特徴色のデータを作成する

# 画像を特徴色に
def img_to_ccolor(input_img):
    # 画像を一つのRGBに
    rgb = img_to_rgb(input_img)
    print("===RGB===")
    print(rgb)
    # RGBを特徴色に
    x = comp_sim(rgb)
    return x

# サンプルの緑茶の画像データ
    ## 濃い綾鷹
    ## 颯
    ## 濃いお〜いお茶
    ## 伊右衛門
    ## 生茶
otya_img_list = ["/content/oiotyakoi.jpeg", "/content/sou.jpeg", "/content/koiayataka.jpg", "/content/iemon.jpeg", "/content/namatya.jpeg"]

otya_ccolor_list = []

for img in otya_img_list:
    print(img)
    ccolor = img_to_ccolor(img)
    print(ccolor)
    otya_ccolor_list.append(ccolor)

print(otya_ccolor_list)

# サンプルの緑茶の味データを作成し、正規化(0~1)した
# サンプルの緑茶の味データを正規化(0~1)
## 濃い綾鷹
## 颯
## 濃いお〜いお茶
## 伊右衛門
## 生茶
tastes = np.array([
    [1,4,4,3],
    [3,5,5,2],
    [1,4,5,2],
    [4,3,4,1],
    [4,1,1,3]
    ])
c = np.linalg.norm(tastes)
normalized_tastes = tastes / c

print(normalized_tastes)

# 表現行列を作成した
# 特徴色から味を予測する表現行列を作成
A = make_hyougen_matrix(normalized_tastes, otya_ccolor_list)


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

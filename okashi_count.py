import streamlit as st
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from torchvision import transforms
from SSDNet import Net

#タイトル、アプリ概要
st.title('お菓子の検出アプリ')
st.subheader('以下のお菓子を検出します')

#検出対象のお菓子
col1, col2, col3  = st.columns(3)

with col1:
    st.write("カントリーマアム")
    st.image("カントリーマアム.jpg", use_column_width=True)

with col2:
    st.write("ブラックサンダー")
    st.image("ブラックサンダー.jpg", use_column_width=True)
    
with col3:
    st.write("アルフォート")
    st.image("アルフォート.jpg", use_column_width=True)   

#入力画像アップロード
uploaded_file = st.file_uploader("ファイルを選択してください", type=[ 'png', 'jpg'])
if uploaded_file is not None:
    i_img=Image.open(uploaded_file)
    st.image(i_img , use_column_width = 'auto')
    #ボタンが押されたら検出モデル稼働
    if st.button(label='お菓子検出'):
        net = Net()
        net = Net(phase='test', num_classes=4).cpu().eval()
        net.load_state_dict(torch.load('ssd2.pt'))
        
        #入力画像をリサイズ＋テンソル化させる関数を定義
        transform = transforms.Compose([
        transforms.Resize((300, 300)),  # 画像のサイズを300x300にリサイズ
        transforms.ToTensor()  # テンソル型に変換
        ])
        #入力画像i_imgをリサイズ＋テンソル化
        x = transform(i_img)
        
        #推論結果yを定義
        y = net(x.unsqueeze(0))

        #検出結果を可視化＋お菓子をカウントする関数を定義
        def visualize_results(input, outputs, threshold):

            img= input.permute(1, 2, 0).numpy()
            image = Image.fromarray((img*255).astype(np.uint8))

            aerial_maritime_labels = ['countrymaam', 'blackthunder', 'alfort']

            scale = torch.Tensor(img.shape[1::-1]).repeat(2)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype('Roboto-Medium.ttf' , 10)
    
            cou = 0
            bla = 0
            alf = 0

            for i in range(outputs.size(1)):
                j = 0
                
                while outputs[0,i,j,0] >= threshold:
                    score = outputs[0,i,j,0]
                    label_name = aerial_maritime_labels[i-1]
                    boxes = (outputs[0,i,j,1:]*scale).cpu().numpy()
                    x0, y0, x1, y1 = draw.textbbox((boxes[0], boxes[1]-10), label_name, font)
                    if label_name == 'countrymaam' :
                        draw.rectangle(boxes, outline='red', width=2)
                        draw.rectangle((x0, y0, x1, y1), fill='red')
                        cou = cou+1
                    elif label_name == 'blackthunder' :
                        draw.rectangle(boxes, outline='yellow', width=2)
                        draw.rectangle((x0, y0, x1, y1), fill='yellow')
                        bla = bla+1
                    else:
                        draw.rectangle(boxes, outline='blue', width=2)
                        draw.rectangle((x0, y0, x1, y1), fill='blue')
                        alf = alf+1
                    
                    draw.text((boxes[0], boxes[1]-10), label_name, font=font, fill='white') 
                    j+=1

            return image , cou , bla , alf
        #関数を実行
        result = visualize_results(x, y, threshold=0.75)

        #検出結果を表示
        st.image(result[0],width = 500 )
        #カウント数を表示します
        okashi_list = ['カントリーマアム','ブラックサンダー','アルフォート']
        zero = 0
        for m in range(len(okashi_list)) :
            if result[m+1] >= 1 :
                st.subheader(f'{okashi_list[m]} : {result[m+1]}個')
                zero = zero+1
        if zero == 0 :
            st.subheader('検知のお菓子は検知されませんでした')
        
        
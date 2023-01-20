import torch
from torchvision import transforms as T
from torchsummary import summary
import json
import streamlit as st
from PIL import Image



# делаем словарь, чтобы по индексу найти название класса
labels = json.load(open('./imagenet_class_index.json'))
# функция декодировки
decode = lambda x: labels[str(x)][1]
print('ok_1')
# загружаем модель
model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v3_small", pretrained=True)
print('ok_2')
resize = T.Resize((224, 224)) 
# подгоняем размер под стандарт модели

print('ok_3')
uploaded_file = st.file_uploader("Choose a file")
print('ok_4')

if uploaded_file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(uploaded_file)
    st.image(uploaded_file, caption='Ваше изображение', use_column_width=True)
    img = img.convert('RGB')
    print('ok_5')
    convert_tensor = T.ToTensor()
    print('ok_6')
    t = convert_tensor(img)
    st.text(t)
    print(t.shape)
    print('ok_7')
    img = resize(t)
    st.text(img.shape)
    print(img.shape)
    model.eval()
    print('ok_9')
    class_title = decode(model(img.unsqueeze(0)).argmax().item())
    st.text(class_title)

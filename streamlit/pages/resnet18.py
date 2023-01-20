import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from torchvision import io
from torchsummary import summary
import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image


# делаем словарь, чтобы по индексу найти название класса
labels = json.load(open('imagenet_class_index.json'))
# функция декодировки
decode = lambda x: labels[str(x)][1]
# загружаем модель
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
resize = T.Resize((224, 224)) 
# подгоняем размер под стандарт модели
uploaded_file = st.file_uploader("Choose a file")


model.fc = nn.Linear(512, 1)
model.load_state_dict(torch.load('resnet18_params.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss()



if uploaded_file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(uploaded_file)
    st.image(uploaded_file, caption='Ваше изображение', use_column_width=True)
    img = img.convert('RGB')
    convert_tensor = T.ToTensor()
    t = convert_tensor(img)
    st.text(t)
    print(t.shape)
    img = resize(t)
    st.text(img.shape)
    print(img.shape)
    model.eval()
    # class_title = decode(model(img.unsqueeze(0)).argmax().item())
    # st.text(class_title)


    model.eval()
    logit = model(img.unsqueeze(0))
    probability = torch.sigmoid(logit)
    class_index = torch.round(probability).item()
    clas_type = ''
    if class_index > 0.95:
        clas_type = 'Dog'
    else:
        clas_type = 'Cat'
    st.text(clas_type)
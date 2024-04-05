from django.shortcuts import render
from django.http import HttpResponse
from .forms import SingleInputFieldForm
import requests
import numpy as np
from keras.utils import load_img, img_to_array
import tensorflow as tf
from io import BytesIO
from PIL import Image
# Create your views here.

def index(request):
    if request.method == 'POST':
        form = SingleInputFieldForm(request.POST)
        if form.is_valid():
            text_value = form.cleaned_data['text_field']
            image_url = text_value
            try:
                response = requests.get(image_url, stream=True)  
                if response.status_code == 200:
                    cnn = tf.keras.models.Sequential()
                    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))
                    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
                    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
                    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
                    cnn.add(tf.keras.layers.Flatten())
                    cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))
                    cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
                    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    cnn.load_weights("/Users/patrickschwarz/Coding/kaggle/blakdetect/chainornot/detect/model.h5")
                    
                    
                    response = requests.get(image_url, stream=True)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    img = img.convert('RGB')
                    img = img.resize((64, 64))
                    img_array = img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0) 
                    result = cnn.predict(img_array)
                    price = result[0][0]*100000
                    return render(request, 'detect/results.html', {'text_value': image_url,'prediction': result[0][0]*100, 'price': price})
                else:
                    # Handle case where image download fails
                    return HttpResponse(f'Failed to download image. (Status code: {response.status_code}')
            except requests.exceptions.RequestException as e:
                # Handle other download errors
                    return HttpResponse('Error downloading image: {e}')
            return render(request, 'detect/results.html', {'text_value': text_value})
    else:
        form = SingleInputFieldForm()
    return render(request, "detect/index.html",{'form': form})
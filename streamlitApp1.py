#!/usr/bin/env python
# coding: utf-8

# In[11]:


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image, ImageOps
import numpy as np


# In[7]:


model = tf.keras.models.load_model('web1ABO_v2.h5')


# In[8]:


st.write("""
         # Fruit Classifier
         """
         )
st.write("This is a simple image classification web app to classify fruits: Apples, Bananas, and Oranges")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


# In[12]:


def import_and_predict(image_data, model):
    
    size = (100,100)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    my_image = image/255
    img_reshape = np.expand_dims(my_image, axis=0)
    prediction = model.predict(img_reshape)
    
    return prediction


# In[13]:


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is an apple!")
        st.write("Usually comes in red and green color")
    elif np.argmax(prediction) == 1:
        st.write("It is a banana!")
    else:
        st.write("It is an orange!")
    
    st.text("Probability (0: Apple, 1: Banana, 2: Orange)")
    st.write(prediction)


# In[ ]:





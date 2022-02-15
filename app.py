import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding',False) ##to avoid erro notification
@st.cache(allow_output_mutation=True) ##to avoid to run all every time
def load_model():
    model = tf.keras.models.load_model('model_vgg19.h5')

    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Tomato_Leaf_Disease_Prediction
         """
         )

file = st.file_uploader("Please upload an Tomato_Leaf image", type=["jpg", "png"])




# st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction

map_dict = {0: 'Tomato___Leaf_Mold',
            1: 'Tomato___Tomato_mosaic_virus',
            2: 'Tomato___Late_blight',
            3: 'Tomato___Early_blight',
            4: 'Tomato___Bacterial_spot',
            5: 'Tomato___Septoria_leaf_spot',
            6: 'Tomato___Spider_mites',
            7: 'spotted_spider_mite',
            8: 'Tomato___Target_Spot',
            9: 'Tomato___healthy',
           10: 'Tomato___Tomato_Yellow_Leaf_Curl_Viruscd'}
 ##string="This image has :"+class_names[np.argmax(predictions)]

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    #class_names=["""Tomato___Bacterial_spot,Tomato___Early_blight,Tomato___healthy,Tomato___Late_blight,
                 #Tomato___Leaf_Mold,Tomato___Septoria_leaf_spot,Tomato___Spider_mites Two-spotted_spider_mite,
                 #Tomato___Target_Spot,Tomato___Tomato_mosaic_virus,Tomato___Tomato_Yellow_Leaf_Curl_Viruscd""" ]
    
    string="Predicted Label for the image is {}".format(map_dict [np.argmax(predictions)])
   
    st.success(string)

 
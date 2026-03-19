import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
#st.write("Versión de Python:", platform.python_version())
image = Image.open('OIG5.jpg')
st.image(image, width=350)
with st.sidebar:
    st.subheader("Usando un modelo entrenado en teachable Machine puedes Usarlo en esta app para identificar")
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:

    # procesamiento imagen...

    prediction = model.predict(data)

    # 👇 AQUÍ VA LO NUEVO
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    respuestas = {
        0: "💖 ¡Eres tú Angie! Qué foto tan linda",
        1: "👀 No eres Angie... pero interesante",
        2: "📱 Veo un celular por ahí",
        3: "🥤 Ese termo se ve útil"
    }

    if confidence > 0.6:
        st.success(respuestas[index])
    else:
        st.warning("🤔 No estoy muy segur@ de lo que veo...")

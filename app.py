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
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
  # Obtener la predicción más alta
index = np.argmax(prediction)
confidence = prediction[0][index]

# Diccionario de respuestas bonitas
respuestas = {
    0: "💖 ¡Eres tú Angie! Qué foto tan linda",
    1: "👀 No eres Angie... pero igual interesante",
    2: "📱 Veo un celular por ahí",
    3: "🥤 Ese termo se ve útil"
}

# Mostrar resultado SOLO si es confiable
if confidence > 0.6:
    st.success(respuestas[index])
else:
    st.warning("🤔 No estoy muy segur@ de lo que veo...")



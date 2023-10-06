import tensorflow as tf
from tensorflow import keras
from base_model import build_base_model
from top_model import build_top_model

top_model=keras.models.load_model("models/top_model.h5")
base_model=keras.models.load_model("models/base_model.h5")

model_input=keras.Input(shape=(128,128,3),name="model_input")
x=base_model(model_input)
model_output=top_model(x)
model=keras.Model(inputs=model_input,outputs=model_output)

model.save('models/EFNB7_Classifier.h5')

""" Mañana correr todo los codigos para revisar que funcione, de preferencia cambiar el numero de epocas que se
    entrena el clasificador (5 por ejemplo). Revisar que el ensamble se haga apropiadamente y finalmente hacer una pequeña app
    en gradio"""


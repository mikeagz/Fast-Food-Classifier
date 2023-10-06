import gradio as gr
import tensorflow as tf
import json
import cv2
import numpy as np

with open('data/class_indices', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)

model=tf.keras.models.load_model("models/EFNB7_Classifier.h5")
labels=list(obj.keys())

example_list=[
    ["data/Ham.jpg"],
    ["data/taco.jpg"],
    ["data/sandwich.jpg"]
]

def predict(path):
    img=cv2.imread(path,1)
    img=cv2.resize(img,(128,128))
    img=np.expand_dims(img,axis=0)
    prediction=model.predict(img,verbose=0)
    confidences = {labels[i]: float(prediction[0][i]) for i in range(10)}  
    return confidences

demo=gr.Interface(fn=predict,
                  inputs=gr.Image(type='filepath'),
                  outputs=gr.Label(),
                  examples=example_list,
                  allow_flagging='never')

demo.launch(server_port=8085)
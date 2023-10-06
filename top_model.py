from keras import models
from keras import layers

def build_top_model():
    Classifier = models.Sequential(name="Top_model")

    Classifier.add(layers.Dense(1024,
                                activation='relu',
                                input_dim=2560))
    Classifier.add(layers.Dropout(0.3))
    Classifier.add(layers.Dense(512,
                                activation='relu',))
    Classifier.add(layers.Dense(10,
                                activation='sigmoid',))
    
    return Classifier


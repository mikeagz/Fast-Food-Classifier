import tensorflow as tf
from tensorflow import keras
import numpy as np
from base_model import build_base_model
import json
import os


def extract_features(model, directory, batch_size, samples):
    features = np.zeros((samples, 2560))
    labels = np.zeros((samples, 10))
    generator = data_gen.flow_from_directory(directory=directory,
                                             target_size=(128, 128),
                                             batch_size=batch_size,
                                             class_mode="categorical")
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = model.predict(inputs_batch,verbose=0)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i += 1
        if i*batch_size >= samples:
            break
    class_index=generator.class_indices
    return features, labels, class_index


if __name__ == "__main__":
    if os.path.exists("models/base_model.h5"):
        EFNB7=keras.models.load_model("models/base_model.h5")
    else:
        EFNB7 = build_base_model()
        EFNB7.save("models/base_model.h5")

    print("Model builded!")

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_features, train_labels , class_indices= extract_features(model=EFNB7,
                                                    directory="data/raw/Fast Food Classification V2/Train",
                                                    batch_size=50,
                                                    samples=7500)
    val_features, val_labels,_ = extract_features(model=EFNB7,
                                                directory="data/raw/Fast Food Classification V2/Valid",
                                                batch_size=50,
                                                samples=2500)
    print("Features Extracted!")

    np.savez("data/processed/train_file",
             features=train_features, 
             labels=train_labels)
    
    np.savez("data/processed/val_file",
             features=val_features, 
             labels=val_labels)
    
    with open("data/class_indices","w") as fp:
        json.dump(class_indices,fp)

    print("Features Saved!")

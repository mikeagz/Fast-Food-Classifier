import tensorflow as tf
from base_model import build_base_model
from top_model import build_top_model
import numpy as np
import matplotlib.pyplot as plt

train = np.load("data/processed/train_file.npz")
train_features, train_labels = train["features"], train["labels"]

val = np.load("data/processed/val_file.npz")
val_features, val_labels = val["features"], val["labels"]

print("Training dataset size (data):", train_features.shape)
print("Training dataset size(labels):", train_labels.shape)


def train_top_model(initial_learning_rate=0.01,
                    decay_steps=1000,
                    alpha=0.001,
                    checkpoint_filepath='tmp/checkpoint',
                    epochs=15,
                    batch_size=32):
    classifier = build_top_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=decay_steps,
        t_mul=1.0,
        m_mul=1.0,
        alpha=alpha
    ))

    classifier.compile(optimizer=optimizer,
                       loss='categorical_crossentropy',
                       metrics=['acc'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                   save_weights_only=True,
                                                                   monitor='val_acc',
                                                                   mode='max',
                                                                   save_best_only=True,
                                                                   initial_value_threshold=0.7)
    learning_rates = []
    learning_rate_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: learning_rates.append(optimizer.lr.numpy()))

    history = classifier.fit(train_features, 
                             train_labels, 
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_data=(val_features, val_labels),
                             callbacks=[model_checkpoint_callback, learning_rate_callback])
    return history,learning_rates,classifier

if __name__=='__main__':
    history,learning_rates,model=train_top_model()

    model.save("models/top_model.h5")

    acc=history.history['acc']
    val_acc=history.history['val_acc']

    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.plot(np.arange(1,16),acc)
    plt.plot(np.arange(1,16),val_acc)
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.legend(["Train","Validation"])
    plt.title("Accuracy Curves")
    plt.grid()

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    plt.subplot(1,3,2)
    plt.plot(np.arange(1,16),loss)
    plt.plot(np.arange(1,16),val_loss)
    plt.xlabel('Epoch')
    plt.legend(["Train","Validation"])
    plt.title("Loss Curves")
    plt.grid()

    plt.subplot(1,3,3)
    plt.plot(range(1, 16), learning_rates)  # Rango de 1 a 10 para las 10 épocas
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Decay with Restarts')
    plt.grid()

    plt.savefig("reports/metrics.png",
                dpi=300,
                bbox_inches='tight')
        
    """ Cargar el checkpoint como mejor modelo; Guardarlo y despues en otro script ensamblar
        el modelo base con el clasificador y probar mediante un request a una imagen.
        Aun no decido si entrenar o correr todo el flujo, pero de ser el caso tambien 
        podría añadir un tracking con wandb. """
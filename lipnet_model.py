import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, GRU, TimeDistributed, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def lipnet_model(vocab_size):
    """
    Define the lipnet model from the paper
    Args:
        vocab_size:

    Returns:

    """
    return Sequential(layers=[
        Conv3D(32, 3, activation='relu', input_shape=(75, 50, 110, 1), padding='same'),
        MaxPooling3D((1, 2, 2)),
        Conv3D(94, 3, activation='relu', padding='same'),
        MaxPooling3D((1, 2, 2)),
        Conv3D(75, 3, activation='relu', padding='same'),
        MaxPooling3D((1, 2, 2)),

        TimeDistributed(Flatten()),

        Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'),
        Dropout(0.5),
        Bidirectional(GRU(128, return_sequences=True, kernel_initializer='Orthogonal'), merge_mode='concat'),
        Dropout(0.5),

        Dense(vocab_size + 1, activation='softmax')
    ]
    )


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def get_scheduler():
    return LearningRateScheduler(scheduler)

# %%
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# %%
class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_end(self, epoch, idx_to_char, logs=None):
        data = self.dataset.as_numpy_iterator().next()
        yhat = self.model.predict(data[0])
        decoded_output = \
        tf.keras.backend.ctc_decode(yhat, input_length=np.ones(yhat.shape[0]) * yhat.shape[1], greedy=False)[0][0]
        for x in range(len(yhat)):
            print("Original: ", tf.strings.reduce_join(idx_to_char(data[1][x]), axis=0).numpy().decode('utf-8'))
            print("Predicted: ", tf.strings.reduce_join(idx_to_char(decoded_output[x]), axis=0).numpy().decode('utf-8'))
            print(" ")


def get_example_producer(dataset):
    return ProduceExample(dataset)
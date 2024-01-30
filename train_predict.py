import cv2
import imageio
import tensorflow as tf
import numpy as np
import gdown
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Union, Any, Optional, Callable
from constants import START_X, START_Y, END_X, END_Y, DATA_URL, VOCAB
from lipnet_model import lipnet_model, CTCLoss, ProduceExample, get_scheduler
from tensorflow.keras.optimizers import Adam


def download_data(data_path: str) -> None:
    url = DATA_URL
    output = 'data.zip'
    gdown.download(url, output, quiet=False)
    gdown.extractall('data.zip')


def load_video(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    iteration_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(iteration_count):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frame = frame[START_Y:END_Y, START_X:END_X, :]
        frames.append(frame)
    cap.release()

    # Standardize frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std
    return frames


VOCABULARY = [char for char in VOCAB]
CHAR_TO_IDX =  tf.keras.layers.StringLookup(
        vocabulary=VOCABULARY,
    )
IDX_TO_CHAR = tf.keras.layers.StringLookup(
        vocabulary=CHAR_TO_IDX.get_vocabulary(), invert=True, mask_token=None
    )


def load_alignments(path: str):
    with open(path, 'r') as f:
        alignments = f.readlines()
    words = []
    for line in alignments:
        line = line.replace('\n', '').split(' ')
        if line[2] != 'sil':
            words.append(" ")
            words.append(line[2])
    tokens = CHAR_TO_IDX(tf.reshape(tf.strings.unicode_split(words, input_encoding='UTF-8'), (-1)))[1:]
    return tokens


def load_data(video_filepath: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    path = bytes.decode(video_filepath.numpy())
    video = load_video(path)
    filename = path.split('/')[-1]
    alignement_path = f"data/alignments/s1/{filename.replace('mpg', 'align')}"
    alignments = load_alignments(alignement_path)
    return video, alignments


def mappable_load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    result = tf.py_function(load_data, [path],  (tf.float32, tf.int64))
    return result


def get_data_split(data_dir: str = './data/s1/*.mpg') -> tf.data.Dataset:
    data = tf.data.Dataset.list_files(data_dir)
    data = data.shuffle(500, reshuffle_each_iteration=False)
    data = data.map(mappable_load_data)
    data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
    data = data.prefetch(tf.data.AUTOTUNE)
    train_data = data.take(50)
    val_data = data.skip(50).take(10)
    test_data = data.skip(60)
    return train_data, val_data, test_data


def train_model(model, epochs: int, data: tf.data.Dataset) -> None:
    model.compile(optimizer=Adam(learning_rate=0.001), loss=CTCLoss)
    schedule_callback = get_scheduler()
    example_callback = ProduceExample(data)
    model.fit(data, epochs=epochs, callbacks=[schedule_callback, example_callback])
    return model


def predict_check_prediction(test_date: tf.data.Dataset, model: tf.keras.Model) -> None:
    test = test_date.as_numpy_iterator().next()
    yhat = model.predict(test[0])
    decoded_output = \
    tf.keras.backend.ctc_decode(yhat, input_length=np.ones(yhat.shape[0]) * yhat.shape[1], greedy=False)[0][0]
    for x in range(len(yhat)):
        print("Original: ", tf.strings.reduce_join(IDX_TO_CHAR(test[1][x]), axis=0).numpy().decode('utf-8'))
        print("Predicted: ", tf.strings.reduce_join(IDX_TO_CHAR(decoded_output[x]), axis=0).numpy().decode('utf-8'))
        print(" ")



if __name__ == "__main__":
    # download_data("data")
    train_data, val_data, test_data = get_data_split()
    model = lipnet_model(len(VOCAB))
    model = train_model(model, 1, train_data)
    predict_check_prediction(test_data, model)


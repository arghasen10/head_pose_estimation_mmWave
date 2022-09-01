import sys
sys.path.append("../")
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from library.augumentation import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
tf.random.set_seed(101)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--loc", default='../static_dataset/processed/data/*', help="Root Path to dataset")
parser.add_argument("-mp", "--model_path", default="./saved_weights/static_saved_weights_cnn.h5",help="Destination to model weights")
parser.add_argument("-c", "--class_count", default=600, help="class_count in balanced dataset")
args = parser.parse_args()

def get_dataset(args):
    data = Dataset(loc=args.loc,
                   class_count=args.class_count,
                   frame_stack=4,
                   dop_min=1375,dop_max=5293)
    lbl_map ={'looking forward': 0,
             'Talking': 1,
             'yawning': 2,
             'looking right': 3,
             'looking left': 4,
             'looking up': 5}

    X_norm = data.data
    y = to_categorical(np.array(list(map(lambda e: lbl_map[e], data.label))), num_classes=6)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def get_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,(5,5),(1,1),padding="same",activation='relu',input_shape=(48,48,4)),
        tf.keras.layers.MaxPool2D((3,3)),
        tf.keras.layers.Conv2D(64,(3,3),(1,1),padding="same",activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(96,(3,3),(1,1),padding="same",activation='relu'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32,"relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(6,"softmax")
    ])
    return model


if __name__=='__main__':
    X_train, X_test, y_train, y_test=get_dataset(args)
    model=get_model()
    model.compile(loss="categorical_crossentropy", optimizer='adam',metrics="accuracy")

    folder=datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    best_save=tf.keras.callbacks.ModelCheckpoint(filepath=args.model_path,save_weights_only=True,
                                                    monitor='val_accuracy',mode='max',save_best_only=True)
    tbd=tf.keras.callbacks.TensorBoard(log_dir=f'logs/{folder}')

    model.fit(
        X_train,
        y_train,
        epochs=10000,
        validation_split=0.2,
        batch_size=32,
        callbacks=[best_save,tbd])



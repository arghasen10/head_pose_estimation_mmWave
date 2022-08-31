import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
from library.augumentation import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append('../')


data = Dataset(loc='../static_dataset/processed/denoised/*',
               class_count=1000,
               frame_stack=4,
               dop_min=1375,
               dop_max=5293, )

lbl_map = \
    {'looking forward': 0,
     'Talking': 1,
     'yawning': 2,
     'looking right': 3,
     'looking left': 4,
     'looking up': 5}

# X=preprocess_input(np.uint8(data.data*255))
X_norm = data.data
y = to_categorical(np.array(list(map(lambda e: lbl_map[e], data.label))), num_classes=6)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), (1, 1), padding="same", activation='relu', input_shape=(48, 48, 4)),
    tf.keras.layers.MaxPool2D((3, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(96, (3, 3), (1, 1), padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, "relu"),
    tf.keras.layers.Dense(6, "softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")

model.summary()

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32)

model.evaluate(X_test, y_test)

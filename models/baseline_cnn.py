import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

# ==========================
# Load Dataset
# ==========================

DATASET = "dataset"

X_train = np.load(os.path.join(DATASET, "X_train.npy"))
X_val = np.load(os.path.join(DATASET, "X_val.npy"))
X_test = np.load(os.path.join(DATASET, "X_test.npy"))

y_train = np.load(os.path.join(DATASET, "y_train.npy"))
y_val = np.load(os.path.join(DATASET, "y_val.npy"))
y_test = np.load(os.path.join(DATASET, "y_test.npy"))

with open(os.path.join(DATASET, "label_mapping.json")) as f:
    label_map = json.load(f)

NUM_CLASSES = len(label_map)

# ==========================
# Prepare Data
# ==========================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print("Training Shape :", X_train.shape)

# ==========================
# CNN Model
# ==========================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=X_train.shape[1:]
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.5),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)

os.makedirs("saved_models", exist_ok=True)

checkpoint = ModelCheckpoint(

    "saved_models/baseline_cnn.keras",

    monitor="val_accuracy",

    save_best_only=True
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = model.fit(

    X_train,

    y_train,

    validation_data=(X_val, y_val),

    epochs=100,

    batch_size=32,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ==========================
# Evaluate
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy :", accuracy)

# ==========================
# Save Curves
# ==========================

os.makedirs("docs/training_curves", exist_ok=True)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")
plt.legend(["Train","Validation"])

plt.subplot(1,2,2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Loss")
plt.legend(["Train","Validation"])

plt.tight_layout()

plt.savefig(
    "docs/training_curves/baseline_cnn_training.png",
    dpi=300
)

plt.show()
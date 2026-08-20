import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================
# Paths
# ==========================

FEATURE_PATH = r"D:\Project\Major\Cry Datasets\Features\MFCC"

OUTPUT_PATH = "dataset"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ==========================
# Read Features
# ==========================

X = []
y = []

classes = sorted(os.listdir(FEATURE_PATH))

label_map = {}

for label, class_name in enumerate(classes):

    label_map[class_name] = label

    class_folder = os.path.join(FEATURE_PATH, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):

        if file.endswith(".npy"):

            feature = np.load(
                os.path.join(class_folder, file)
            )

            X.append(feature)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Features Shape :", X.shape)
print("Labels Shape   :", y.shape)

# ==========================
# Normalize
# ==========================

samples = X.shape[0]

X_flat = X.reshape(samples, -1)

scaler = StandardScaler()

X_flat = scaler.fit_transform(X_flat)

X = X_flat.reshape(X.shape)

# Save scaler
joblib.dump(
    scaler,
    os.path.join(OUTPUT_PATH, "scaler.pkl")
)

# ==========================
# Split Dataset
# ==========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# ==========================
# Save Dataset
# ==========================

np.save(os.path.join(OUTPUT_PATH, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_PATH, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_PATH, "X_test.npy"), X_test)

np.save(os.path.join(OUTPUT_PATH, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_PATH, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_PATH, "y_test.npy"), y_test)

with open(
    os.path.join(OUTPUT_PATH, "label_mapping.json"),
    "w"
) as f:
    json.dump(label_map, f, indent=4)

print("\n================================")
print("Dataset Preparation Complete!")
print("================================")

print("Training Samples :", len(X_train))
print("Validation Samples :", len(X_val))
print("Testing Samples :", len(X_test))
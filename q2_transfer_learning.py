"""
Question 2: Transfer Learning - Freeze vs. Fine-Tune
CS5720 - Home Assignment 3
Student: Palash Palit

Compares two ways of reusing a ResNet50 pretrained on ImageNet on the
TensorFlow flowers dataset (5 classes):
  Experiment A - Feature extraction: freeze the whole base, train only a new classifier.
  Experiment B - Fine-tuning: unfreeze the last conv block (conv5) and train it with the classifier.
Both run for the same number of epochs; the script reports trainable parameters,
training time, validation accuracy, a training-loss plot, and a comparison table.

Recommended: run on Google Colab with a T4 GPU (see HomeAssignment3_Colab.ipynb).
"""

# Setup: imports, GPU check, and settings shared by both experiments
import os, time, tarfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

print("TensorFlow version:", tf.__version__)
print("GPU available:", bool(tf.config.list_physical_devices("GPU")))

tf.keras.utils.set_random_seed(42)   # makes the runs repeatable
IMG_SIZE = (224, 224)                 # ResNet50's standard input size
BATCH_SIZE = 32
EPOCHS = 5                            # same number of epochs for both experiments
WEIGHTS = "imagenet"                  # pretrained ImageNet weights

# Load the flowers dataset (downloads automatically, about 220 MB)
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file("flower_photos.tgz", origin=DATA_URL)
data_root = os.path.join(os.path.dirname(archive), "flowers_extracted")
if not os.path.exists(os.path.join(data_root, "flower_photos")):
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(data_root, filter="data")   # safe extraction (Python 3.12+)
        except TypeError:
            tar.extractall(data_root)                  # older Python versions
DATA_DIR = os.path.join(data_root, "flower_photos")

# 80% training / 20% validation split; each subfolder is one class
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ResNet50 expects its own preprocessing (channel order and mean subtraction)
def prep(images, labels):
    return preprocess_input(images), labels

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

def build_model(fine_tune):
    """Pretrained ResNet50 base + a new classifier for our classes.

    fine_tune=False -> Experiment A: the whole base is frozen.
    fine_tune=True  -> Experiment B: the last conv block (conv5_*) is unfrozen.
    """
    base = ResNet50(include_top=False,            # drop the original 1000-class ImageNet classifier
                    weights=WEIGHTS,
                    input_shape=IMG_SIZE + (3,),
                    pooling="avg")                # global average pooling -> 2048 features

    if not fine_tune:
        base.trainable = False                    # freeze every layer in the base
    else:
        base.trainable = True
        for layer in base.layers:
            # Unfreeze only the last convolutional block (conv5). BatchNorm layers stay
            # frozen so their pretrained statistics are not disturbed by our small dataset.
            layer.trainable = (layer.name.startswith("conv5_")
                               and not isinstance(layer, layers.BatchNormalization))

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)              # keep BatchNorm in inference mode
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)   # new classification layer
    return models.Model(inputs, outputs)


def count_params(model):
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    frozen = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return trainable, frozen


def run_experiment(name, fine_tune, learning_rate):
    model = build_model(fine_tune)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    trainable, frozen = count_params(model)
    print(f"\n=== {name} ===")
    print(f"Trainable parameters: {trainable:,}   Frozen parameters: {frozen:,}")

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    train_time = time.time() - start

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Training time: {train_time:.1f} s   Validation accuracy: {val_acc:.4f}")
    return {"name": name, "trainable": trainable, "frozen": frozen,
            "time": train_time, "val_acc": val_acc, "val_loss": val_loss,
            "history": history.history}

# Only the new Dense layer trains, so a normal learning rate (1e-3) is used.
result_a = run_experiment("Frozen Feature Extractor", fine_tune=False, learning_rate=1e-3)

# Pretrained conv5 weights are updated too, so a smaller learning rate (1e-4) is used
# to adjust them gently instead of destroying what they already learned.
result_b = run_experiment("Fine-Tuned Network", fine_tune=True, learning_rate=1e-4)

epochs_range = range(1, EPOCHS + 1)
plt.figure(figsize=(7, 4.5))
plt.plot(epochs_range, result_a["history"]["loss"], marker="o", label="Frozen feature extractor")
plt.plot(epochs_range, result_b["history"]["loss"], marker="s", label="Fine-tuned network")
plt.xlabel("Epoch"); plt.ylabel("Training loss")
plt.title("Training Loss: Frozen vs. Fine-Tuned ResNet50")
plt.xticks(list(epochs_range)); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout()
plt.savefig("training_loss_comparison.png", dpi=150)
plt.show()  # also saved as training_loss_comparison.png

table = pd.DataFrame({
    "Method": [result_a["name"], result_b["name"]],
    "Trainable Parameters": [f"{result_a['trainable']:,}", f"{result_b['trainable']:,}"],
    "Training Time (s)": [f"{result_a['time']:.1f}", f"{result_b['time']:.1f}"],
    "Validation Accuracy": [f"{result_a['val_acc']:.4f}", f"{result_b['val_acc']:.4f}"],
})
print(table.to_string(index=False))

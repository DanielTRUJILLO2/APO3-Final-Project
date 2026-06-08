import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8
SEED = 42
EPOCHS = 30


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS_DIR / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        SPLITS_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(100, seed=SEED).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_history(history):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(REPORTS_DIR / "cnn_history.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("CNN accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_accuracy.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("CNN loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cnn_loss.png")
    plt.close()


def get_predictions(model, test_ds):
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)

    ax.set_title("Matriz de confusion - CNN")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_cnn.png")
    plt.close()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Cargando datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets()

    print("Clases detectadas:")
    print(class_names)

    print("Construyendo CNN...")
    model = build_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        )
    ]

    print("Entrenando CNN...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Guardando modelo...")
    model.save(MODELS_DIR / "cnn_model.keras")

    print("Evaluando CNN...")
    y_true, y_pred = get_predictions(model, test_ds)

    print()
    print("=" * 60)
    print("CNN")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        "model": "CNN",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "cnn_results.csv", index=False)

    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, class_names)

    print()
    print("Modelo CNN guardado en models/cnn_model.keras")
    print("Resultados guardados en reports/cnn_results.csv")
    print("Graficas guardadas en reports/figures/")


if __name__ == "__main__":
    main()

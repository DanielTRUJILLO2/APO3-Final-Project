from pathlib import Path
import os
import sys
import joblib
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from features import extract_features


MODELS_DIR = Path("models")
IMAGE_SIZE = (128, 128)

CNN_CLASS_NAMES = ["bueno", "malo", "regular"]


def load_ml_model(model_name="random_forest"):
    model_path = MODELS_DIR / f"{model_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    return joblib.load(model_path)


def load_cnn_model():
    import tensorflow as tf

    model_path = MODELS_DIR / "cnn_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    return tf.keras.models.load_model(model_path)


def predict_with_ml(image_path, model_name="random_forest"):
    model = load_ml_model(model_name)
    features = extract_features(image_path).reshape(1, -1)

    prediction = str(model.predict(features)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    return prediction, confidence


def predict_with_cnn(image_path):
    model = load_cnn_model()

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    array = np.array(image)
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    index = int(np.argmax(probabilities))

    prediction = CNN_CLASS_NAMES[index]
    confidence = float(probabilities[index])

    return prediction, confidence


def estimate_size(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))

    array = np.array(image).astype(np.float32)

    brightness = np.mean(array, axis=2)
    saturation = np.max(array, axis=2) - np.min(array, axis=2)

    center_region = array[64:192, 64:192]
    border_pixels = np.concatenate([
        array[:20, :, :].reshape(-1, 3),
        array[-20:, :, :].reshape(-1, 3),
        array[:, :20, :].reshape(-1, 3),
        array[:, -20:, :].reshape(-1, 3),
    ])

    background_color = np.median(border_pixels, axis=0)

    color_distance = np.linalg.norm(array - background_color, axis=2)

    mask = (color_distance > 35) | (saturation > 45)

    mask[:5, :] = False
    mask[-5:, :] = False
    mask[:, :5] = False
    mask[:, -5:] = False

    area_ratio = float(np.mean(mask))

    if area_ratio < 0.12:
        size = "pequeño"
    elif area_ratio < 0.28:
        size = "mediano"
    else:
        size = "grande"

    return size, area_ratio


def predict_image(image_path, model_type="random_forest"):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    if model_type == "cnn":
        quality, confidence = predict_with_cnn(image_path)
    elif model_type == "svm":
        quality, confidence = predict_with_ml(image_path, "svm")
    else:
        quality, confidence = predict_with_ml(image_path, "random_forest")

    size, area_ratio = estimate_size(image_path)

    return {
        "quality": quality,
        "confidence": confidence,
        "size": size,
        "area_ratio": area_ratio,
        "model": model_type
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/predict.py ruta_imagen modelo")
        print("Modelos disponibles: random_forest, svm, cnn")
        sys.exit(1)

    image_path = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) >= 3 else "random_forest"

    result = predict_image(image_path, model_type=model_type)

    print()
    print("Resultado de prediccion")
    print("-----------------------")
    print(f"Modelo: {result['model']}")
    print(f"Calidad: {result['quality']}")
    print(f"Tamano: {result['size']}")
    print(f"Area relativa: {result['area_ratio']:.2f}")

    if result["confidence"] is not None:
        print(f"Confianza: {result['confidence']:.2f}")

from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


IMAGE_SIZE = (128, 128)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return image


def color_histogram(array, bins=16):
    features = []

    for channel in range(3):
        hist, _ = np.histogram(
            array[:, :, channel],
            bins=bins,
            range=(0, 255),
            density=True
        )
        features.extend(hist)

    return np.array(features)


def basic_color_features(array):
    features = []

    for channel in range(3):
        channel_data = array[:, :, channel]
        features.append(np.mean(channel_data))
        features.append(np.std(channel_data))
        features.append(np.min(channel_data))
        features.append(np.max(channel_data))

    return np.array(features)


def dark_area_features(array):
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]

    brightness = np.mean(array, axis=2)

    dark_pixels = brightness < 70
    brown_dark_pixels = (red < 120) & (green < 90) & (blue < 80)

    return np.array([
        np.mean(dark_pixels),
        np.mean(brown_dark_pixels),
        np.mean(brightness),
        np.std(brightness)
    ])


def edge_features(image):
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)

    return np.array([
        np.mean(edge_array),
        np.std(edge_array),
        np.mean(edge_array > 40)
    ])


def extract_features(image_path):
    image = load_image(image_path)
    array = np.array(image).astype(np.float32)

    features = np.concatenate([
        basic_color_features(array),
        color_histogram(array),
        dark_area_features(array),
        edge_features(image)
    ])

    return features

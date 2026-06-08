from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features


SPLITS_DIR = Path("data/splits")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

CLASSES = ["bueno", "regular", "malo"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_split(split_name):
    X = []
    y = []

    for class_name in CLASSES:
        class_dir = SPLITS_DIR / split_name / class_name

        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                features = extract_features(image_path)
                X.append(features)
                y.append(class_name)

    return np.array(X), np.array(y)


def plot_confusion_matrix(y_true, y_pred, model_name):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix)

    ax.set_title(f"Matriz de confusión - {model_name}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()

    output_path = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(output_path)
    plt.close()


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(y_test, y_pred, labels=CLASSES, zero_division=0))

    plot_confusion_matrix(y_test, y_pred, model_name)

    return metrics


def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4]
    }

    model = RandomForestClassifier(random_state=42, class_weight="balanced")

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Mejores parámetros Random Forest:")
    print(grid.best_params_)

    return grid.best_estimator_


def train_svm(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=42))
    ])

    param_grid = {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["linear", "rbf"],
        "svm__gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Mejores parámetros SVM:")
    print(grid.best_params_)

    return grid.best_estimator_


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Cargando datos...")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print(f"Train: {X_train.shape[0]}")
    print(f"Val: {X_val.shape[0]}")
    print(f"Test: {X_test.shape[0]}")
    print(f"Características por imagen: {X_train.shape[1]}")

    print()
    print("Entrenando Random Forest...")
    random_forest = train_random_forest(X_train_full, y_train_full)

    print()
    print("Entrenando SVM...")
    svm = train_svm(X_train_full, y_train_full)

    joblib.dump(random_forest, MODELS_DIR / "random_forest.pkl")
    joblib.dump(svm, MODELS_DIR / "svm.pkl")

    print()
    print("Evaluando modelos...")

    results = []
    results.append(evaluate_model(random_forest, X_test, y_test, "Random Forest"))
    results.append(evaluate_model(svm, X_test, y_test, "SVM"))

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORTS_DIR / "ml_results.csv", index=False)

    print()
    print("Resultados guardados en reports/ml_results.csv")
    print("Modelos guardados en models/")
    print("Matrices de confusión guardadas en reports/figures/")


if __name__ == "__main__":
    main()


from pathlib import Path
import shutil
import csv
import sys

from predict import predict_image


TEST_DIR = Path("data/splits/test")
OUTPUT_DIR = Path("reports/misclassified")
REPORT_PATH = Path("reports/misclassified_report.csv")

CLASSES = ["bueno", "regular", "malo"]


def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "svm"

    output_model_dir = OUTPUT_DIR / model_type

    if output_model_dir.exists():
        shutil.rmtree(output_model_dir)

    output_model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    correct = 0

    for real_class in CLASSES:
        folder = TEST_DIR / real_class

        for image_path in folder.iterdir():
            if not image_path.is_file():
                continue

            result = predict_image(image_path, model_type=model_type)

            predicted = result["quality"]
            confidence = result["confidence"]

            total += 1

            if predicted == real_class:
                correct += 1
            else:
                error_folder = output_model_dir / f"real_{real_class}_pred_{predicted}"
                error_folder.mkdir(parents=True, exist_ok=True)

                destination = error_folder / image_path.name
                shutil.copy2(image_path, destination)

                rows.append({
                    "model": model_type,
                    "image": image_path.name,
                    "real": real_class,
                    "predicted": predicted,
                    "confidence": confidence,
                    "copied_to": str(destination)
                })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "image", "real", "predicted", "confidence", "copied_to"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Modelo: {model_type}")
    print(f"Accuracy: {correct}/{total} = {correct / total:.2f}")
    print(f"Errores: {len(rows)}")
    print(f"Reporte guardado en: {REPORT_PATH}")
    print(f"Imagenes copiadas en: {output_model_dir}")


if __name__ == "__main__":
    main()

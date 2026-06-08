from pathlib import Path
import csv
import random
import shutil


SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SOURCE_DIR = Path("data/processed/balanced_raw")
SPLITS_DIR = Path("data/splits")
METADATA_PATH = Path("data/metadata.csv")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

QUALITY_MAP = {
    "good": "bueno",
    "regular": "regular",
    "bad": "malo"
}


def clean_previous_outputs():
    if SPLITS_DIR.exists():
        shutil.rmtree(SPLITS_DIR)

    for split in ["train", "val", "test"]:
        for quality in ["bueno", "regular", "malo"]:
            (SPLITS_DIR / split / quality).mkdir(parents=True, exist_ok=True)

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_leaf_folders(root):
    return [
        folder for folder in root.rglob("*")
        if folder.is_dir() and not any(child.is_dir() for child in folder.iterdir())
    ]


def parse_folder_name(folder_name):
    parts = folder_name.split("_")

    fruit = parts[0].lower()

    if len(parts) < 2:
        raise ValueError(f"No se pudo leer la calidad desde la carpeta: {folder_name}")

    quality_raw = parts[-1].lower()

    if quality_raw not in QUALITY_MAP:
        raise ValueError(f"Calidad no reconocida en la carpeta: {folder_name}")

    quality = QUALITY_MAP[quality_raw]

    return fruit, quality, quality_raw


def split_files(files):
    files = list(files)
    random.shuffle(files)

    n = len(files)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    return {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }


def main():
    random.seed(SEED)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta fuente: {SOURCE_DIR}")

    clean_previous_outputs()

    rows = []

    leaf_folders = get_leaf_folders(SOURCE_DIR)

    for folder in leaf_folders:
        fruit, quality, quality_raw = parse_folder_name(folder.name)

        files = [
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        ]

        if not files:
            continue

        splits = split_files(files)

        for split_name, split_files_list in splits.items():
            for file in split_files_list:
                destination = SPLITS_DIR / split_name / quality / file.name
                shutil.copy2(file, destination)

                rows.append({
                    "filename": file.name,
                    "fruit": fruit,
                    "quality": quality,
                    "quality_original": quality_raw,
                    "split": split_name,
                    "source_path": str(file),
                    "split_path": str(destination)
                })

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "filename",
            "fruit",
            "quality",
            "quality_original",
            "split",
            "source_path",
            "split_path"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Dataset preparado correctamente")
    print(f"Metadata guardada en: {METADATA_PATH}")
    print(f"Splits guardados en: {SPLITS_DIR}")
    print(f"Total de imagenes procesadas: {len(rows)}")

    for split in ["train", "val", "test"]:
        print()
        print(split.upper())
        for quality in ["bueno", "regular", "malo"]:
            count = len(list((SPLITS_DIR / split / quality).glob("*")))
            print(f"{quality}: {count}")


if __name__ == "__main__":
    main()


import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


# =========================
# 1. CAU HINH
# =========================
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
INITIAL_LR = 1e-4

CLASS_NAMES = [
    "with_mask",
    "without_mask",
    "with_incorrect_mask",
]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3-class face mask detector using MobileNetV2"
    )
    parser.add_argument(
        "--dataset", type=str, default="dataset",
        help="Dataset folder containing one subfolder per class"
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Folder to save model, reports and plots"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Maximum number of training epochs"
    )
    return parser.parse_args()


# =========================
# 2. DOC DANH SACH ANH
# =========================
def collect_images(dataset_dir):
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay thu muc dataset: {dataset_dir.resolve()}"
        )

    image_paths = []
    labels = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Thieu thu muc lop: {class_dir.resolve()}"
            )

        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]

        if len(files) < 10:
            raise ValueError(
                f"Lop {class_name} chi co {len(files)} anh. "
                "Hay bo sung it nhat 10 anh de chay thu."
            )

        image_paths.extend(str(p.resolve()) for p in files)
        labels.extend([class_id] * len(files))

        print(f"[INFO] {class_name}: {len(files)} images")

    return np.array(image_paths), np.array(labels, dtype=np.int32)


# =========================
# 3. TAO TAP TRAIN / VAL / TEST
# =========================
def split_dataset(paths, labels):
    train_val_paths, test_paths, train_val_y, test_y = (
        train_test_split(
            paths,
            labels,
            test_size=0.15,
            random_state=SEED,
            stratify=labels,
        )
    )

    # 15% tong du lieu cho validation:
    # 0.15 / 0.85 = 0.17647 tren phan con lai
    train_paths, val_paths, train_y, val_y = train_test_split(
        train_val_paths,
        train_val_y,
        test_size=0.17647,
        random_state=SEED,
        stratify=train_val_y,
    )

    print(f"\n[INFO] Train: {len(train_paths)} images")
    print(f"[INFO] Validation: {len(val_paths)} images")
    print(f"[INFO] Test: {len(test_paths)} images")

    return (
        (train_paths, train_y),
        (val_paths, val_y),
        (test_paths, test_y),
    )


# =========================
# 4. DOC VA TIEN XU LY ANH
# =========================
def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)

    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=2,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


# =========================
# 5. XAY DUNG MOBILENETV2
# =========================
def build_model():
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )

    inputs = layers.Input(shape=(224, 224, 3))

    # Augmentation chi hoat dong khi training
    x = augmentation(inputs)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )

    # Giai doan dau: dong bang backbone
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Ba lop, moi lop co mot xac suat
    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="mask_classification",
    )(x)

    model = Model(inputs, outputs, name="mask_detector_v2")

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =========================
# 6. HUAN LUYEN VA DANH GIA
# =========================
def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading image paths...")
    image_paths, labels = collect_images(args.dataset)

    # Luu thu tu nhan de dung lai khi du doan
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)

    train_data, val_data, test_data = split_dataset(
        image_paths, labels
    )

    train_ds = make_dataset(*train_data, training=True)
    val_ds = make_dataset(*val_data)
    test_ds = make_dataset(*test_data)

    print("[INFO] Building MobileNetV2...")
    model = build_model()
    model.summary()

    model_path = output_dir / "mask_detector.keras"

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("[INFO] Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Nap lai checkpoint tot nhat truoc khi test
    model = tf.keras.models.load_model(model_path)

    print("\n[INFO] Evaluating on held-out test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    y_true = np.concatenate(
        [y.numpy() for _, y in test_ds],
        axis=0,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    print("\nClassification report:")
    print(report)

    with open(
        output_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    print("Confusion matrix (true rows, predicted columns):")
    print(cm)

    with open(output_dir / "confusion_matrix.txt", "w") as f:
        f.write(np.array2string(cm))

    # Ve bieu do training
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train accuracy")
    axes[1].plot(
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    axes[1].set_title("Training and validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_plot.png", dpi=150)
    plt.close(fig)

    print(f"\n[INFO] Model saved at: {model_path.resolve()}")
    print(f"[INFO] Labels saved at: {(output_dir / 'labels.json').resolve()}")
    print(f"[INFO] Reports and plots saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

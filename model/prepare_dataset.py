
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import shutil

# Chỉnh các đường dẫn này nếu thư mục của bạn khác
ROOT = Path(__file__).resolve().parent
RAW_IMAGES = ROOT / "data_raw" / "images"
RAW_XML = ROOT / "data_raw" / "annotations"
OUTPUT = ROOT / "dataset"

# Ánh xạ nhãn từ Larxel sang tên lớp trong file train
CLASS_MAP = {
    "with_mask": "with_mask",
    "without_mask": "without_mask",
    "mask_weared_incorrect": "with_incorrect_mask",
}

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]


def find_image(filename, xml_path):
    candidates = [RAW_IMAGES / filename]
    candidates += [
        RAW_IMAGES / f"{xml_path.stem}{ext}"
        for ext in IMAGE_EXTENSIONS
    ]

    for path in candidates:
        if path.is_file():
            return path

    return None


def main():
    if not RAW_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay thu muc anh: {RAW_IMAGES}"
        )

    if not RAW_XML.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay thu muc XML: {RAW_XML}"
        )

    # Tạo lại dataset đầu ra để tránh lẫn dữ liệu cũ
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for class_name in CLASS_MAP.values():
        (OUTPUT / class_name).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in CLASS_MAP.values()}
    skipped = 0

    xml_files = list(RAW_XML.rglob("*.xml"))
    print(f"[INFO] Tim thay {len(xml_files)} file XML")

    for xml_path in xml_files:
        try:
            root = ET.parse(xml_path).getroot()
            filename_node = root.find("filename")

            filename = (
                filename_node.text.strip()
                if filename_node is not None and filename_node.text
                else xml_path.stem
            )

            image_path = find_image(filename, xml_path)

            if image_path is None:
                print(f"[WARN] Khong tim thay anh: {filename}")
                skipped += 1
                continue

            with Image.open(image_path) as source:
                image = source.convert("RGB")
                width, height = image.size

                for index, obj in enumerate(root.findall("object")):
                    name_node = obj.find("name")
                    box = obj.find("bndbox")

                    if name_node is None or box is None:
                        continue

                    original_label = (
                        name_node.text.strip()
                        if name_node.text
                        else ""
                    )

                    if original_label not in CLASS_MAP:
                        continue

                    coords = {}
                    valid = True

                    for key in ("xmin", "ymin", "xmax", "ymax"):
                        node = box.find(key)
                        if node is None or node.text is None:
                            valid = False
                            break
                        try:
                            coords[key] = int(float(node.text))
                        except ValueError:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Giới hạn bounding box trong phạm vi ảnh
                    xmin = max(0, min(width, coords["xmin"]))
                    ymin = max(0, min(height, coords["ymin"]))
                    xmax = max(0, min(width, coords["xmax"]))
                    ymax = max(0, min(height, coords["ymax"]))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    # Thêm khoảng đệm 10% quanh khuôn mặt
                    pad_x = int((xmax - xmin) * 0.10)
                    pad_y = int((ymax - ymin) * 0.10)

                    xmin = max(0, xmin - pad_x)
                    ymin = max(0, ymin - pad_y)
                    xmax = min(width, xmax + pad_x)
                    ymax = min(height, ymax + pad_y)

                    face = image.crop((xmin, ymin, xmax, ymax))
                    class_name = CLASS_MAP[original_label]

                    # Tên duy nhất, tránh ghi đè ảnh khác
                    output_name = (
                        f"{image_path.stem}_obj{index}.jpg"
                    )
                    save_path = OUTPUT / class_name / output_name
                    face.save(save_path, quality=95)

                    counts[class_name] += 1

        except (ET.ParseError, OSError, ValueError) as error:
            print(f"[WARN] Loi doc {xml_path.name}: {error}")
            skipped += 1

    print("\n===== KET QUA CHUAN BI DATASET =====")
    for class_name, count in counts.items():
        print(f"{class_name}: {count} anh")

    print(f"XML bo qua/loi: {skipped}")
    print(f"Dataset dau ra: {OUTPUT.resolve()}")

    if any(count < 10 for count in counts.values()):
        print(
            "[WARN] Co lop duoi 10 anh. "
            "Hay kiem tra annotation va du lieu."
        )


if __name__ == "__main__":
    main()

import argparse
import csv
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image


ALLOWED_GENDERS = {"Men", "Women", "Unisex"}
ALLOWED_MASTER_CATEGORIES = {"Apparel", "Footwear", "Accessories"}
OUTPUT_COLUMNS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "usage",
    "productDisplayName",
    "sourceDataset",
    "imagePath",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and materialize a filtered local fashion seed dataset from Hugging Face."
    )
    parser.add_argument(
        "--dataset",
        default="Transformersx/fashion-product-images-small",
        help="Hugging Face dataset id to download.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the filtered manifest and image files should be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on exported rows after filtering. Use 0 for no cap.",
    )
    return parser.parse_args()


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def normalize_value(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def export_dataset(dataset_name: str, split: str, output_dir: Path, limit: int) -> None:
    dataset = load_dataset(dataset_name, split=split)

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "seed_manifest.csv"

    exported = 0
    skipped = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row in dataset:
            gender = normalize_value(row.get("gender"))
            master_category = normalize_value(row.get("masterCategory"))
            image = row.get("image")

            if gender not in ALLOWED_GENDERS or master_category not in ALLOWED_MASTER_CATEGORIES or image is None:
                skipped += 1
                continue

            product_id = normalize_value(row.get("id"))
            if not product_id:
                skipped += 1
                continue

            image_path = image_dir / f"{product_id}.jpg"
            ensure_rgb(image).save(image_path, format="JPEG", quality=92)

            writer.writerow(
                {
                    "id": product_id,
                    "gender": gender.lower(),
                    "masterCategory": master_category.lower(),
                    "subCategory": normalize_value(row.get("subCategory")).lower(),
                    "articleType": normalize_value(row.get("articleType")).lower(),
                    "baseColour": normalize_value(row.get("baseColour")).lower(),
                    "season": normalize_value(row.get("season")).lower(),
                    "usage": normalize_value(row.get("usage")).lower(),
                    "productDisplayName": normalize_value(row.get("productDisplayName")),
                    "sourceDataset": dataset_name,
                    "imagePath": image_path.name,
                }
            )

            exported += 1
            if limit > 0 and exported >= limit:
                break

            if exported % 500 == 0:
                print(f"Exported {exported} images...", flush=True)

    print(f"Export complete. Exported={exported}, Skipped={skipped}, Output={output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    export_dataset(args.dataset, args.split, Path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()

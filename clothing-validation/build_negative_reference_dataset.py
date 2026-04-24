import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torchvision.datasets import Caltech256, Flowers102, Food101, OxfordIIITPet


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "clothing-validation" / "non_clothing"
DEFAULT_DOWNLOAD_ROOT = REPO_ROOT / "data" / "clothing-validation" / ".downloads"

CALTECH_EXCLUDED_KEYWORDS = {
    "t-shirt",
    "shirt",
    "shoe",
    "skirt",
    "trousers",
    "jeans",
    "bra",
    "briefs",
    "dress",
    "jacket",
    "coat",
    "sweater",
}


@dataclass(frozen=True)
class SampleTask:
    group_name: str
    count: int


def save_selected_images(dataset, indices: list[int], output_dir: Path, prefix: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for order, index in enumerate(indices, start=1):
        image, _ = dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        output_path = output_dir / f"{prefix}_{order:04d}.jpg"
        if output_path.exists():
            continue

        image.convert("RGB").save(output_path, format="JPEG", quality=95)
        saved += 1

    return saved


def choose_indices(total: int, desired: int, rng: random.Random) -> list[int]:
    desired = min(total, desired)
    return sorted(rng.sample(range(total), desired))


def choose_caltech_indices(dataset: Caltech256, desired: int, rng: random.Random) -> list[int]:
    allowed_indices = []
    for index, category_id in enumerate(dataset.y):
        category_name = dataset.categories[category_id]
        normalized = category_name.lower()
        if any(keyword in normalized for keyword in CALTECH_EXCLUDED_KEYWORDS):
            continue
        allowed_indices.append(index)

    desired = min(len(allowed_indices), desired)
    return sorted(rng.sample(allowed_indices, desired))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pets-count", type=int, default=600)
    parser.add_argument("--food-count", type=int, default=400)
    parser.add_argument("--flowers-count", type=int, default=400)
    parser.add_argument("--objects-count", type=int, default=800)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = args.output_root.resolve()
    download_root = args.download_root.resolve()

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        SampleTask("pets", args.pets_count),
        SampleTask("food", args.food_count),
        SampleTask("flowers", args.flowers_count),
        SampleTask("objects", args.objects_count),
    ]

    print("Preparing curated non-clothing reference dataset...")
    print(f"Output root: {output_root}")
    print(f"Download cache: {download_root}")

    pets = OxfordIIITPet(root=str(download_root), split="trainval", target_types="category", download=True)
    pets_saved = save_selected_images(
        pets,
        choose_indices(len(pets), tasks[0].count, rng),
        output_root / tasks[0].group_name,
        "pet",
    )
    print(f"Saved {pets_saved} pet negatives.")

    food = Food101(root=str(download_root), split="train", download=True)
    food_saved = save_selected_images(
        food,
        choose_indices(len(food), tasks[1].count, rng),
        output_root / tasks[1].group_name,
        "food",
    )
    print(f"Saved {food_saved} food negatives.")

    flowers = Flowers102(root=str(download_root), split="train", download=True)
    flowers_saved = save_selected_images(
        flowers,
        choose_indices(len(flowers), tasks[2].count, rng),
        output_root / tasks[2].group_name,
        "flower",
    )
    print(f"Saved {flowers_saved} flower negatives.")

    objects = Caltech256(root=str(download_root), download=True)
    objects_saved = save_selected_images(
        objects,
        choose_caltech_indices(objects, tasks[3].count, rng),
        output_root / tasks[3].group_name,
        "object",
    )
    print(f"Saved {objects_saved} general-object negatives.")

    total_saved = pets_saved + food_saved + flowers_saved + objects_saved
    print(f"Curated non-clothing dataset ready. Saved {total_saved} images in total.")


if __name__ == "__main__":
    main()

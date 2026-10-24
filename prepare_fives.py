import argparse
import shutil
from pathlib import Path

from PIL import Image

from data_utils import IMAGE_EXTENSIONS, discover_pairs


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the extracted official FIVES dataset into this project's layout."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Root of the extracted official FIVES archive containing train/ and test/.",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalized_name(path):
    return "".join(character for character in path.name.lower() if character.isalnum())


def locate_directory(source, split, role):
    split = split.lower()
    role = role.lower()
    matches = []
    for path in source.rglob("*"):
        if not path.is_dir() or path.parent.name.lower() != split:
            continue
        name = normalized_name(path)
        if role == "images" and name == "original":
            matches.append(path)
        if role == "masks" and name == "groundtruth":
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {split}/{role} directory under {source}, found: {matches}"
        )
    return matches[0]


def index_images(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.lower()
        if key in result:
            raise ValueError(f"Duplicate sample name '{key}' in {directory}")
        result[key] = path
    return result


def copy_split(source, destination, split, expected_count, overwrite):
    source_images = locate_directory(source, split, "images")
    source_masks = locate_directory(source, split, "masks")
    images = index_images(source_images)
    masks = index_images(source_masks)
    if set(images) != set(masks):
        raise ValueError(
            f"FIVES {split} image/mask mismatch: "
            f"missing masks={sorted(set(images) - set(masks))[:10]}, "
            f"missing images={sorted(set(masks) - set(images))[:10]}"
        )
    if len(images) != expected_count:
        raise ValueError(
            f"Expected {expected_count} FIVES {split} pairs, found {len(images)}. "
            "Confirm that the official archive was fully extracted."
        )

    destination_images = destination / f"{split}_images"
    destination_masks = destination / f"{split}_masks"
    for directory in (destination_images, destination_masks):
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            raise FileExistsError(f"Destination is not empty: {directory}. Use --overwrite.")
        directory.mkdir(parents=True, exist_ok=True)

    for key in sorted(images):
        shutil.copy2(images[key], destination_images / images[key].name)
        shutil.copy2(masks[key], destination_masks / masks[key].name)

    first_key = sorted(images)[0]
    with Image.open(images[first_key]) as image, Image.open(masks[first_key]) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {first_key}: {image.size} versus {mask.size}"
            )
    print(f"Prepared FIVES {split}: {len(images)} pairs")


def main():
    args = parse_args()
    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Extracted FIVES directory not found: {source}")
    destination = args.data_root.resolve() / "FIVES"
    copy_split(source, destination, "train", 600, args.overwrite)
    copy_split(source, destination, "test", 200, args.overwrite)

    train_pairs = discover_pairs(args.data_root, "FIVES", "train")
    test_pairs = discover_pairs(args.data_root, "FIVES", "test")
    print(f"Verification complete: train={len(train_pairs)}, test={len(test_pairs)}")
    print(f"Prepared dataset: {destination}")


if __name__ == "__main__":
    main()

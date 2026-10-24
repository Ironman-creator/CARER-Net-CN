import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".gif"}
DATASET_CHOICES = ("DRIVE", "STARE", "CHASE", "FIVES")


def sample_key(path, dataset):
    """Return a stable image/mask key for every supported dataset."""
    stem = Path(path).stem.lower()
    dataset = dataset.upper()
    patterns = {
        "DRIVE": r"^(\d+)",
        "STARE": r"^(im\d+)",
        "CHASE": r"^(image_\d+[lr])",
        "FIVES": r"^(.+)$",
    }
    if dataset not in patterns:
        raise ValueError(f"Unsupported dataset '{dataset}'. Available: {', '.join(DATASET_CHOICES)}")
    match = re.search(patterns[dataset], stem)
    if not match:
        raise ValueError(f"Cannot derive a {dataset} sample key from '{Path(path).name}'")
    return match.group(1)


def _index_files(directory, dataset):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    index = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = sample_key(path, dataset)
        if key in index:
            raise ValueError(f"Duplicate sample key '{key}' in {directory}")
        index[key] = path
    return index


def discover_pairs(data_root, dataset, split):
    dataset = dataset.upper()
    base = Path(data_root) / dataset
    images = _index_files(base / f"{split}_images", dataset)
    masks = _index_files(base / f"{split}_masks", dataset)

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Image/mask mismatch for {dataset} {split}: "
            f"missing masks={missing_masks}, missing images={missing_images}"
        )
    if not images:
        raise ValueError(f"No image/mask pairs found for {dataset} {split}")

    return [(images[key], masks[key], key) for key in sorted(images)]


def discover_auxiliary(directory, dataset):
    """Index optional files such as official field-of-view masks by sample key."""
    return _index_files(directory, dataset.upper())


def split_train_validation(pairs, validation_ratio, seed):
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_ratio))
    if validation_count >= len(indices):
        raise ValueError("Validation split leaves no training samples")
    validation_indices = set(indices[:validation_count])
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in validation_indices]
    validation_pairs = [pair for index, pair in enumerate(pairs) if index in validation_indices]
    return train_pairs, validation_pairs


def _augment_pair(image, mask):
    if random.random() < 0.5:
        image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
    if random.random() < 0.5:
        image, mask = ImageOps.flip(image), ImageOps.flip(mask)

    angle = random.choice((0, 90, 180, 270))
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)

    if random.random() < 0.4:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.4:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.3:
        gamma = random.uniform(0.8, 1.2)
        lut = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
        image = image.point(lut * 3 if image.mode == "RGB" else lut)
    return image, mask


class RetinalDataset(Dataset):
    def __init__(self, pairs, image_size=512, input_mode="grayscale", augment=False):
        self.pairs = list(pairs)
        self.image_size = (image_size, image_size)
        self.input_mode = input_mode
        self.augment = augment
        if input_mode not in {"grayscale", "green", "gray_rgb"}:
            raise ValueError("input_mode must be 'grayscale', 'green', or 'gray_rgb'")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path, key = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)

        if self.augment:
            image, mask = _augment_pair(image, mask)

        rgb_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.input_mode == "green":
            image_array = rgb_array[:, :, 1]
            image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0)
        elif self.input_mode == "gray_rgb":
            # Keep the exact PIL grayscale channel used by the existing GN
            # baseline and append RGB cues for baseline-preserving adapters.
            grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            image_array = np.concatenate((grayscale[:, :, None], rgb_array), axis=2)
            image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1)
        else:
            image_array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0)

        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_array.copy()).unsqueeze(0)
        return image_tensor, mask_tensor, key


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

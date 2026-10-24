import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm", ".gif"}
DATASET_CHOICES = ("DRIVE", "STARE", "CHASE", "FIVES")


def clahe_uint8(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply deterministic CLAHE to a two-dimensional uint8 image.

    The implementation uses clipped per-tile histograms followed by bilinear
    interpolation between tile lookup tables.  Keeping this implementation in
    the repository avoids an OpenCV dependency and makes the preprocessing
    identical on the training and evaluation computers.
    """
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("clahe_uint8 expects a two-dimensional grayscale image")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    grid_y, grid_x = map(int, tile_grid_size)
    height, width = image.shape
    if grid_y < 1 or grid_x < 1 or grid_y > height or grid_x > width:
        raise ValueError("tile_grid_size must fit inside the input image")

    y_edges = np.linspace(0, height, grid_y + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, grid_x + 1, dtype=np.int32)
    lookup_tables = np.empty((grid_y, grid_x, 256), dtype=np.uint8)

    for tile_y in range(grid_y):
        for tile_x in range(grid_x):
            tile = image[
                y_edges[tile_y] : y_edges[tile_y + 1],
                x_edges[tile_x] : x_edges[tile_x + 1],
            ]
            histogram = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
            clip_count = max(1, int(round(clip_limit * tile.size / 256.0)))
            excess = int(np.maximum(histogram - clip_count, 0).sum())
            histogram = np.minimum(histogram, clip_count)
            if excess:
                histogram += excess // 256
                remainder = excess % 256
                if remainder:
                    # Spread the remainder across the full intensity range
                    # instead of biasing it toward dark bins.
                    indices = np.floor(
                        np.arange(remainder, dtype=np.float64) * 256.0 / remainder
                    ).astype(np.int32)
                    histogram[indices] += 1

            cumulative = histogram.cumsum(dtype=np.int64)
            nonzero = np.flatnonzero(cumulative)
            if not len(nonzero) or cumulative[-1] == cumulative[nonzero[0]]:
                lookup_tables[tile_y, tile_x] = np.arange(256, dtype=np.uint8)
                continue
            minimum = cumulative[nonzero[0]]
            mapped = np.rint(
                (cumulative - minimum) * 255.0 / (cumulative[-1] - minimum)
            )
            lookup_tables[tile_y, tile_x] = np.clip(mapped, 0, 255).astype(np.uint8)

    y_centers = (y_edges[:-1] + y_edges[1:] - 1) / 2.0
    x_centers = (x_edges[:-1] + x_edges[1:] - 1) / 2.0

    def interpolation_indices(length, centers):
        coordinates = np.arange(length, dtype=np.float64)
        upper = np.searchsorted(centers, coordinates, side="right")
        upper = np.clip(upper, 0, len(centers) - 1)
        lower = np.clip(upper - 1, 0, len(centers) - 1)
        denominator = centers[upper] - centers[lower]
        weight = np.zeros(length, dtype=np.float64)
        valid = denominator > 0
        weight[valid] = (
            coordinates[valid] - centers[lower[valid]]
        ) / denominator[valid]
        return lower, upper, np.clip(weight, 0.0, 1.0)

    y0, y1, wy = interpolation_indices(height, y_centers)
    x0, x1, wx = interpolation_indices(width, x_centers)
    values = image
    top_left = lookup_tables[y0[:, None], x0[None, :], values].astype(np.float32)
    top_right = lookup_tables[y0[:, None], x1[None, :], values].astype(np.float32)
    bottom_left = lookup_tables[y1[:, None], x0[None, :], values].astype(np.float32)
    bottom_right = lookup_tables[y1[:, None], x1[None, :], values].astype(np.float32)
    wx = wx[None, :]
    wy = wy[:, None]
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return np.rint(top * (1.0 - wy) + bottom * wy).astype(np.uint8)


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
        if input_mode not in {"grayscale", "green", "gray_rgb", "gray_rgb_clahe"}:
            raise ValueError(
                "input_mode must be 'grayscale', 'green', 'gray_rgb', or "
                "'gray_rgb_clahe'"
            )

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
        elif self.input_mode in {"gray_rgb", "gray_rgb_clahe"}:
            # Keep the exact PIL grayscale channel used by the existing GN
            # baseline as channel zero.  Evaluation derives its FOV from this
            # unmodified channel even when the tested model consumes CLAHE.
            grayscale_uint8 = np.asarray(image.convert("L"), dtype=np.uint8)
            grayscale = grayscale_uint8.astype(np.float32) / 255.0
            channels = [grayscale[:, :, None], rgb_array]
            if self.input_mode == "gray_rgb_clahe":
                clahe = clahe_uint8(
                    grayscale_uint8, clip_limit=2.0, tile_grid_size=(8, 8)
                ).astype(np.float32) / 255.0
                channels.append(clahe[:, :, None])
            image_array = np.concatenate(channels, axis=2)
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

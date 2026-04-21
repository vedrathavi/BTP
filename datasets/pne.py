import copy
import os
import random
from typing import List, Tuple

from torchvision import datasets, transforms


def _iid_split(num_samples: int, num_clients: int, seed: int) -> List[List[int]]:
    """Split indices into reproducible IID partitions for simulated FL clients."""
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    splits = []
    chunk = max(num_samples // num_clients, 1)
    for i in range(num_clients):
        start = i * chunk
        end = (i + 1) * chunk if i < num_clients - 1 else num_samples
        splits.append(indices[start:end])
    return splits


def _merge_imagefolder_datasets(base_dataset, extra_dataset):
    """Merge two torchvision ImageFolder datasets into a single shallow-copied dataset.

    Why this exists:
        This project trains on train+val while keeping test strictly held out.
    How it helps:
        Reuses ImageFolder metadata without custom dataset reimplementation.
    """
    merged = copy.copy(base_dataset)
    merged.samples = list(base_dataset.samples) + list(extra_dataset.samples)
    merged.targets = list(base_dataset.targets) + list(extra_dataset.targets)
    if hasattr(base_dataset, "imgs"):
        merged.imgs = list(base_dataset.imgs) + list(extra_dataset.imgs)
    return merged


def _resolve_roots(data_dir: str):
    """Resolve train/val/test roots from common pneumonia dataset layouts."""
    candidates = [
        (os.path.join(data_dir, "train"), os.path.join(data_dir, "val"), os.path.join(data_dir, "test")),
        (os.path.join(data_dir, "dataset-pne", "train"), os.path.join(data_dir, "dataset-pne", "val"), os.path.join(data_dir, "dataset-pne", "test")),
    ]
    for train_root, val_root, test_root in candidates:
        if os.path.isdir(train_root) and os.path.isdir(val_root) and os.path.isdir(test_root):
            return train_root, val_root, test_root
    raise FileNotFoundError("Could not find dataset-pne folders. Expected train/, val/, test/")


def load_data(data_dir: str, image_size: int, num_clients: int = 4, seed: int = 42) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, List[List[int]]]:
    """Load dataset-pne from train/val/test folders.

    Training uses train + val combined; test uses the dedicated test folder.
    """
    train_root, val_root, test_root = _resolve_roots(data_dir)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_root, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_root, transform=eval_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx or train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("dataset-pne class folders do not match across train/val/test")

    # Combine train and validation for local-client training, keep test untouched.
    train_dataset = _merge_imagefolder_datasets(train_dataset, val_dataset)
    client_splits = _iid_split(len(train_dataset), num_clients, seed)
    return train_dataset, test_dataset, client_splits
import csv
import os
import random
from typing import List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


ALL_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def label_from_findings(findings: str) -> List[int]:
    labels = [item.strip() for item in str(findings).split("|") if item.strip()]
    return [1 if disease in labels else 0 for disease in ALL_DISEASES]


def _balanced_subsample_multilabel(samples: Sequence[Tuple[str, List[int]]], max_samples_per_class: int, seed: int) -> List[Tuple[str, List[int]]]:
    class_indices = {i: [] for i in range(len(ALL_DISEASES))}
    for idx, (_, label_vec) in enumerate(samples):
        for class_idx, value in enumerate(label_vec):
            if value == 1:
                class_indices[class_idx].append(idx)

    rng = random.Random(seed)
    selected_indices = set()
    for indices in class_indices.values():
        rng.shuffle(indices)
        selected_indices.update(indices[:max_samples_per_class])

    ordered_indices = sorted(selected_indices)
    return [samples[i] for i in ordered_indices]


def _iid_split(num_samples: int, num_clients: int, seed: int) -> List[List[int]]:
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


def _read_name_list(file_path: str) -> Optional[set]:
    if not os.path.isfile(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return set(names)


class NIHMultiLabelDataset(Dataset):
    """Multi-label NIH chest X-ray dataset with 14 disease targets."""

    def __init__(self, image_root: str, csv_path: str, file_names: Optional[Sequence[str]] = None, transform=None):
        self.root = image_root
        self.csv_path = csv_path
        self.transform = transform
        self.samples = []
        self.targets = []
        self.classes = list(ALL_DISEASES)
        self.class_to_idx = {name: idx for idx, name in enumerate(ALL_DISEASES)}

        file_name_set = set(file_names) if file_names is not None else None

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"NIH metadata CSV not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_name = row.get("Image Index", "").strip()
                if not image_name:
                    continue
                if file_name_set is not None and image_name not in file_name_set:
                    continue

                image_path = os.path.join(image_root, image_name)

                finding_labels = row.get("Finding Labels", "").strip()
                target = label_from_findings(finding_labels)
                self.samples.append((image_path, target))
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.float32)


def _resolve_roots(data_dir: str):
    image_candidates = [
        os.path.join(data_dir, "images-224", "images-224"),
        os.path.join(data_dir, "images-224"),
        os.path.join(data_dir, "images"),
        data_dir,
    ]
    for image_root in image_candidates:
        if os.path.isdir(image_root):
            csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
            train_list = os.path.join(data_dir, "train_val_list_NIH.txt")
            test_list = os.path.join(data_dir, "test_list_NIH.txt")
            return image_root, csv_path, train_list, test_list
    raise FileNotFoundError("Could not find NIH image root")


def load_data(
    data_dir: str,
    image_size: int,
    num_clients: int = 4,
    seed: int = 42,
    max_samples_per_class: int = 2000,
) -> Tuple[NIHMultiLabelDataset, NIHMultiLabelDataset, List[List[int]]]:
    """Load NIH chest X-ray data from CSV metadata and official split files."""
    image_root, csv_path, train_list_path, test_list_path = _resolve_roots(data_dir)

    train_names = _read_name_list(train_list_path)
    test_names = _read_name_list(test_list_path)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = NIHMultiLabelDataset(image_root=image_root, csv_path=csv_path, file_names=train_names, transform=train_transform)
    test_dataset = NIHMultiLabelDataset(image_root=image_root, csv_path=csv_path, file_names=test_names, transform=eval_transform)

    train_dataset.samples = _balanced_subsample_multilabel(train_dataset.samples, max_samples_per_class=max_samples_per_class, seed=seed)
    train_dataset.targets = [label for _, label in train_dataset.samples]

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("NIH dataset split is empty. Check the CSV and split list files.")

    client_splits = _iid_split(len(train_dataset), num_clients, seed)
    return train_dataset, test_dataset, client_splits

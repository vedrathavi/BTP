import argparse
import copy
import json
import math
import os
import random
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from algorithms.adaptive_fedavg import aggregate as adaptive_fedavg_aggregate
from algorithms.fedavg import aggregate as fedavg_aggregate
from datasets.nih import load_data as load_nih_data
from datasets.pne import load_data as load_pne_data

ImageFile.LOAD_TRUNCATED_IMAGES = True

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_ROOT = os.path.join(BASE_DIR, "outputs")
CURRENT_RUN_DIR = os.path.join(OUTPUTS_ROOT, "current_run")
HISTORY_DIR = os.path.join(OUTPUTS_ROOT, "history")
PLOTS_DIR = os.path.join(CURRENT_RUN_DIR, "plots")
LOGS_DIR = os.path.join(CURRENT_RUN_DIR, "logs")

# Imbalance and aggregation safety controls
EVAL_THRESHOLD = 0.25

NUM_WORKERS = 2 if torch.cuda.is_available() else 0
PIN_MEMORY = bool(torch.cuda.is_available())


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Pneumonia Training (menu based)")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root path")
    parser.add_argument("--num-clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=20, help="Communication rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs per client")
    parser.add_argument("--batch-size", type=int, default=32, help="Local and test batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="",
        choices=["", "fedavg", "adaptive_fedavg"],
        help="Optional non-interactive algorithm selection",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["", "nih", "pne"],
        help="Optional non-interactive dataset selection",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def archive_current_run():
    os.makedirs(OUTPUTS_ROOT, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    if not os.path.isdir(CURRENT_RUN_DIR):
        return

    existing_items = os.listdir(CURRENT_RUN_DIR)
    if not existing_items:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = os.path.join(HISTORY_DIR, f"run_{stamp}")
    os.makedirs(archived, exist_ok=True)

    for name in existing_items:
        shutil.move(os.path.join(CURRENT_RUN_DIR, name), os.path.join(archived, name))

    print(f"Archived previous run to: {archived}")


def ensure_run_dirs():
    os.makedirs(CURRENT_RUN_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def choose_menu(title, options):
    print(f"\n{title}")
    for idx, (_, label) in enumerate(options, start=1):
        print(f"{idx}. {label}")

    while True:
        raw = input("Enter choice number: ").strip()
        if raw.isdigit():
            choice = int(raw)
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
        print("Invalid choice. Please enter a valid number.")


class PneumoniaCNN(nn.Module):
    def __init__(self, in_channels=3, num_outputs=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, num_outputs))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def create_model(num_outputs=1):
    return PneumoniaCNN(in_channels=3, num_outputs=num_outputs)


def get_loader_from_indices(dataset, indices, batch_size, shuffle_train=False):
    ds = copy.copy(dataset)
    ds.samples = [dataset.samples[i] for i in indices]
    ds.targets = [dataset.targets[i] for i in indices]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def local_train(
    model,
    dataloader,
    device,
    epochs=1,
    lr=1e-3,
    weight_decay=1e-4,
    round_idx=None,
    client_idx=None,
):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    total_batches = 0
    epoch_desc = "Loader"
    if client_idx is not None and round_idx is not None:
        epoch_desc = f"R{round_idx} C{client_idx} loader"

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0
        batch_bar = tqdm(dataloader, desc=f"{epoch_desc} e{epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for imgs, ys in batch_bar:
            imgs = imgs.to(device, non_blocking=PIN_MEMORY)
            ys = ys.float().to(device, non_blocking=PIN_MEMORY)
            if ys.ndim == 1:
                ys = ys.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_loss = float(loss.item())
            running_loss += batch_loss
            total_batches += 1
            epoch_loss += batch_loss
            epoch_batches += 1

            batch_bar.set_postfix(loss=f"{batch_loss:.4f}")

        scheduler.step()

    avg_loss = running_loss / max(total_batches, 1)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, avg_loss


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_prob = []

    for imgs, ys in dataloader:
        imgs = imgs.to(device, non_blocking=PIN_MEMORY)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        ys_np = ys.numpy()
        if probs.ndim == 1:
            probs = probs[:, None]
        if ys_np.ndim == 1:
            ys_np = ys_np[:, None]
        y_prob.append(probs)
        y_true.append(ys_np)

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = (y_prob >= EVAL_THRESHOLD).astype(int)

    is_multilabel = y_true.shape[1] > 1

    if is_multilabel:
        accuracy = float((y_true == y_pred).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        try:
            auc = float(roc_auc_score(y_true, y_prob, average="macro"))
        except Exception:
            auc = float("nan")
        specificity = float("nan")
        balanced_accuracy = float("nan")
        tn = fp = fn = tp = -1
    else:
        y_true_flat = y_true.reshape(-1)
        y_prob_flat = y_prob.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        tn = int(np.sum((y_true_flat == 0) & (y_pred_flat == 0)))
        fp = int(np.sum((y_true_flat == 0) & (y_pred_flat == 1)))
        fn = int(np.sum((y_true_flat == 1) & (y_pred_flat == 0)))
        tp = int(np.sum((y_true_flat == 1) & (y_pred_flat == 1)))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat,
            y_pred_flat,
            average="binary",
            zero_division=0,
        )
        specificity = tn / max(tn + fp, 1)
        balanced_accuracy = 0.5 * (recall + specificity)

        try:
            auc = roc_auc_score(y_true_flat, y_prob_flat) if len(np.unique(y_true_flat)) > 1 else float("nan")
        except Exception:
            auc = float("nan")

        accuracy = float(accuracy_score(y_true_flat, y_pred_flat))

    sensitivity = recall

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "balanced_accuracy": float(balanced_accuracy),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_true": y_true,
        "y_prob": y_prob,
    }


def compute_weight_drift(global_prev, global_new):
    sq_sum = 0.0
    numel = 0
    for k in global_prev.keys():
        if global_prev[k].dtype == torch.float32:
            diff = (global_new[k].cpu() - global_prev[k].cpu()).float()
            sq_sum += float(torch.sum(diff * diff).item())
            numel += diff.numel()
    if numel == 0:
        return 0.0
    return float(math.sqrt(sq_sum / numel))


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Final Global Model")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_multilabel_roc_curve(y_true, y_prob, class_names, save_path):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("Multi-label ROC plotting expects 2D arrays")

    plt.figure(figsize=(10, 8))

    valid_class_count = 0
    for class_idx, class_name in enumerate(class_names):
        class_true = y_true[:, class_idx]
        class_prob = y_prob[:, class_idx]
        if len(np.unique(class_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(class_true, class_prob)
        class_auc = roc_auc_score(class_true, class_prob)
        plt.plot(fpr, tpr, linewidth=1.5, alpha=0.8, label=f"{class_name} (AUC={class_auc:.3f})")
        valid_class_count += 1

    if valid_class_count == 0:
        plt.close()
        return

    micro_fpr, micro_tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    micro_auc = roc_auc_score(y_true, y_prob, average="micro")
    plt.plot(micro_fpr, micro_tpr, color="black", linewidth=3, label=f"micro-average (AUC={micro_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-label ROC Curve - Final Global Model")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(tn, fp, fn, tp, save_path, title):
    matrix = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12, fontweight="bold")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def run():
    args = parse_args()
    set_seed(args.seed)

    algorithm_options = [
        ("fedavg", "FedAvg"),
        ("adaptive_fedavg", "Adaptive FedAvg"),
    ]
    dataset_options = [
        ("nih", "dataset-nih"),
        ("pne", "dataset-pne"),
    ]

    selected_algorithm = args.algorithm if args.algorithm else choose_menu("Select Algorithm:", algorithm_options)
    selected_dataset = args.dataset if args.dataset else choose_menu("Select Dataset:", dataset_options)

    dataset_loaders = {
        "nih": load_nih_data,
        "pne": load_pne_data,
    }

    requested_data_dir = os.path.abspath(args.data_dir)
    fallback_dirs = {
        "nih": os.path.join(BASE_DIR, "dataset-nih"),
        "pne": os.path.join(BASE_DIR, "dataset-pne"),
    }

    if os.path.isdir(requested_data_dir):
        data_dir = requested_data_dir
    elif os.path.basename(requested_data_dir).lower() == "dataset" and os.path.isdir(fallback_dirs[selected_dataset]):
        data_dir = fallback_dirs[selected_dataset]
    else:
        expected_default = fallback_dirs[selected_dataset]
        raise FileNotFoundError(
            f"Dataset directory not found: {requested_data_dir}. "
            f"For {selected_dataset}, expected folder like: {expected_default}"
        )

    archive_current_run()
    ensure_run_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Algorithm: {selected_algorithm}")
    print(f"Dataset: {selected_dataset}")
    print(f"Data directory: {data_dir}")

    train_dataset, test_dataset, client_splits = dataset_loaders[selected_dataset](
        data_dir=data_dir,
        image_size=args.image_size,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    if len(client_splits) != args.num_clients:
        raise ValueError("Dataset loader returned an unexpected number of client splits")

    print("Class mapping:", train_dataset.class_to_idx)
    print("Client sizes:", [len(s) for s in client_splits])

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    num_outputs = len(getattr(train_dataset, "classes", []))
    if selected_dataset == "nih":
        num_outputs = 14
    elif num_outputs <= 2:
        num_outputs = 1

    if selected_dataset == "nih":
        print("Sample label example:", train_dataset.samples[0][1])
        print("Total samples after subsampling:", len(train_dataset.samples))

    global_model = create_model(num_outputs=num_outputs).to(device)
    global_weights = copy.deepcopy(global_model.state_dict())

    global_round_rows = []
    client_round_rows = []
    weight_drift_rows = []
    client_test_acc_history = {cid: [] for cid in range(args.num_clients)}

    adaptive_config = {
        "beta_size": 0.4,
        "beta_perf": 0.6,
        "temperature": 2.0,
        "min_client_weight": 1e-3,
    }

    for rnd in range(1, args.num_rounds + 1):
        print(f"\n=== Round {rnd}/{args.num_rounds} ===")

        for cid in range(args.num_clients):
            client_test_acc_history[cid].append(np.nan)

        local_weights = []
        local_sizes = []
        local_performances = []

        prev_global_cpu = {k: v.detach().cpu().clone() for k, v in global_weights.items()}

        for cid in tqdm(range(args.num_clients), desc=f"Round {rnd} clients", leave=False, dynamic_ncols=True):
            client_indices = client_splits[cid]
            if not client_indices:
                continue

            train_loader = get_loader_from_indices(
                dataset=train_dataset,
                indices=client_indices,
                batch_size=args.batch_size,
                shuffle_train=True,
            )

            local_model = create_model(num_outputs=num_outputs).to(device)
            local_model.load_state_dict(global_weights)

            updated_weights, train_loss = local_train(
                local_model,
                train_loader,
                device,
                epochs=args.local_epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                round_idx=rnd,
                client_idx=cid,
            )

            local_metrics = evaluate_model(local_model, test_loader, device)
            local_acc = local_metrics["accuracy"]
            local_perf = local_metrics["f1"]

            local_weights.append(updated_weights)
            local_sizes.append(len(client_indices))
            local_performances.append(local_perf)

            client_test_acc_history[cid][-1] = local_acc

            client_round_rows.append(
                {
                    "round": rnd,
                    "client": cid,
                    "n_train_samples": len(client_indices),
                    "local_train_loss": train_loss,
                    "local_test_accuracy": local_metrics["accuracy"],
                    "local_test_precision": local_metrics["precision"],
                    "local_test_recall": local_metrics["recall"],
                    "local_test_f1": local_metrics["f1"],
                    "local_test_auc": local_metrics["auc"],
                    "local_test_specificity": local_metrics["specificity"],
                    "local_test_sensitivity": local_metrics["sensitivity"],
                    "local_test_balanced_accuracy": local_metrics["balanced_accuracy"],
                    "size_weight": np.nan,
                    "performance_weight": np.nan,
                    "adaptive_weight": np.nan,
                }
            )

            print(
                f"Client {cid} -> loss: {train_loss:.4f}, acc: {local_metrics['accuracy']:.4f}, "
                f"prec: {local_metrics['precision']:.4f}, rec: {local_metrics['recall']:.4f}, "
                f"spec: {local_metrics['specificity']:.4f}, sens: {local_metrics['sensitivity']:.4f}, "
                f"bal_acc: {local_metrics['balanced_accuracy']:.4f}, f1: {local_metrics['f1']:.4f}, "
                f"auc: {local_metrics['auc']:.4f}, cm:[tn={local_metrics['tn']}, fp={local_metrics['fp']}, fn={local_metrics['fn']}, tp={local_metrics['tp']}]"
            )

        if not local_weights:
            print("No client updates generated in this round.")
            continue

        active_clients = list(range(len(local_weights)))
        aggregation_weights = local_weights
        aggregation_sizes = local_sizes
        aggregation_performances = local_performances

        if selected_algorithm == "fedavg":
            new_global_cpu, details = fedavg_aggregate(aggregation_weights, aggregation_sizes)
        else:
            new_global_cpu, details = adaptive_fedavg_aggregate(
                local_weights=aggregation_weights,
                local_sizes=aggregation_sizes,
                local_performances=aggregation_performances,
                config=adaptive_config,
            )

        selected_rows = [r for r in client_round_rows if r["round"] == rnd]
        for out_idx, client_idx in enumerate(active_clients):
            for row in selected_rows:
                if row["client"] == client_idx:
                    row["size_weight"] = details["size_weights"][out_idx]
                    row["performance_weight"] = details["performance_weights"][out_idx]
                    row["adaptive_weight"] = details["adaptive_weights"][out_idx]

        global_weights = {k: v.to(device) for k, v in new_global_cpu.items()}
        global_model.load_state_dict(global_weights)

        drift_l2 = compute_weight_drift(prev_global_cpu, new_global_cpu)
        weight_drift_rows.append({"round": rnd, "global_weight_drift_l2": drift_l2})

        global_metrics = evaluate_model(global_model, test_loader, device)
        global_round_rows.append(
            {
                "round": rnd,
                "global_accuracy": global_metrics["accuracy"],
                "global_precision": global_metrics["precision"],
                "global_recall": global_metrics["recall"],
                "global_f1": global_metrics["f1"],
                "global_auc": global_metrics["auc"],
                "global_specificity": global_metrics["specificity"],
                "global_sensitivity": global_metrics["sensitivity"],
                "global_balanced_accuracy": global_metrics["balanced_accuracy"],
                "global_tn": global_metrics["tn"],
                "global_fp": global_metrics["fp"],
                "global_fn": global_metrics["fn"],
                "global_tp": global_metrics["tp"],
                "mean_client_accuracy": float(np.nanmean([r["local_test_accuracy"] for r in selected_rows])),
                "std_client_accuracy": float(np.nanstd([r["local_test_accuracy"] for r in selected_rows])),
                "global_weight_drift_l2": drift_l2,
            }
        )

        print(
            f"Global -> acc: {global_metrics['accuracy']:.4f}, prec: {global_metrics['precision']:.4f}, "
            f"rec: {global_metrics['recall']:.4f}, spec: {global_metrics['specificity']:.4f}, "
            f"sens: {global_metrics['sensitivity']:.4f}, bal_acc: {global_metrics['balanced_accuracy']:.4f}, "
            f"f1: {global_metrics['f1']:.4f}, auc: {global_metrics['auc']:.4f}, "
            f"cm:[tn={global_metrics['tn']}, fp={global_metrics['fp']}, fn={global_metrics['fn']}, tp={global_metrics['tp']}], "
            f"drift: {drift_l2:.6f}"
        )

    if not global_round_rows:
        raise RuntimeError("Training produced no global rounds. Check dataset and split sizes.")

    final_test_eval = evaluate_model(global_model, test_loader, device)

    global_df = pd.DataFrame(global_round_rows)
    client_df = pd.DataFrame(client_round_rows)
    drift_df = pd.DataFrame(weight_drift_rows)

    global_df.to_csv(os.path.join(CURRENT_RUN_DIR, "metrics.csv"), index=False)
    client_df.to_csv(os.path.join(LOGS_DIR, "client_round_metrics.csv"), index=False)
    drift_df.to_csv(os.path.join(LOGS_DIR, "weight_drift.csv"), index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": selected_algorithm,
        "dataset": selected_dataset,
        "config": {
            "data_dir": data_dir,
            "num_clients": args.num_clients,
            "num_rounds": args.num_rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": str(device),
        },
        "final_metrics": {
            "accuracy": float(final_test_eval["accuracy"]),
            "f1": float(final_test_eval["f1"]),
            "auc": float(final_test_eval["auc"]),
            "precision": float(final_test_eval["precision"]),
            "recall": float(final_test_eval["recall"]),
            "specificity": float(final_test_eval["specificity"]),
            "balanced_accuracy": float(final_test_eval["balanced_accuracy"]),
        },
        "best_global_accuracy": float(np.nanmax(global_df["global_accuracy"].values)),
        "best_global_f1": float(np.nanmax(global_df["global_f1"].values)),
        "best_global_auc": float(np.nanmax(global_df["global_auc"].values)),
    }

    with open(os.path.join(CURRENT_RUN_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(global_df["round"], global_df["global_accuracy"], marker="o", linewidth=2, label="Global Accuracy")
    plt.plot(global_df["round"], global_df["global_f1"], marker="s", linewidth=2, label="Global F1")
    plt.plot(global_df["round"], global_df["global_auc"], marker="^", linewidth=2, label="Global AUC")
    plt.plot(global_df["round"], global_df["global_balanced_accuracy"], marker="d", linewidth=2, label="Balanced Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Metric Value")
    plt.title("Global Metrics vs Rounds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "global_metrics_vs_rounds.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(drift_df["round"], drift_df["global_weight_drift_l2"], marker="o", linewidth=2)
    plt.xlabel("Communication Round")
    plt.ylabel("L2 Drift")
    plt.title("Global Weight Drift vs Rounds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "weight_drift_vs_rounds.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cid in range(args.num_clients):
        client_series = [row["local_test_accuracy"] for row in client_round_rows if row["client"] == cid]
        plt.plot(range(1, len(client_series) + 1), client_series, linestyle="--", alpha=0.7, label=f"Client {cid}")
    plt.plot(global_df["round"], global_df["global_accuracy"], color="black", linewidth=3, label="Global")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Client and Global Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "client_convergence_effect.png"), dpi=150)
    plt.close()

    if selected_dataset == "nih":
        plot_multilabel_roc_curve(
            final_test_eval["y_true"],
            final_test_eval["y_prob"],
            class_names=getattr(train_dataset, "classes", []),
            save_path=os.path.join(PLOTS_DIR, "roc_curve_global.png"),
        )
    elif final_test_eval["y_true"].ndim == 2 and final_test_eval["y_true"].shape[1] == 1 and len(np.unique(final_test_eval["y_true"])) > 1:
        plot_roc_curve(
            final_test_eval["y_true"].reshape(-1),
            final_test_eval["y_prob"].reshape(-1),
            save_path=os.path.join(PLOTS_DIR, "roc_curve_global.png"),
        )

    if final_test_eval["tn"] >= 0:
        plot_confusion_matrix(
            final_test_eval["tn"],
            final_test_eval["fp"],
            final_test_eval["fn"],
            final_test_eval["tp"],
            save_path=os.path.join(PLOTS_DIR, "confusion_matrix_global.png"),
            title="Global Model Confusion Matrix",
        )

    print("\nRun complete.")
    print(f"Saved run directory: {CURRENT_RUN_DIR}")
    print(f"Summary file: {os.path.join(CURRENT_RUN_DIR, 'summary.json')}")
    print(f"Metrics file: {os.path.join(CURRENT_RUN_DIR, 'metrics.csv')}")


if __name__ == "__main__":
    run()

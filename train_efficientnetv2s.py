"""
NutriGenus - EfficientNetV2-S transfer learning for vitamin deficiency classification
=======================================================================================

Trains EfficientNetV2-S (ImageNet-pretrained) on the vitamin-deficiency facial/skin
image dataset, EXCLUDING the "Retina Blood Vessel" class folder.

Dataset source (download separately, this script does not fetch it):
https://www.kaggle.com/datasets/ujjwalsinha01/vitamin-deficiency-and-retinal-analysis-image-dataset

Expected folder layout after download/extraction:

    data_dir/
        Vitamin A/       *.jpg / *.jpeg / *.png
        Vitamin B/
        Vitamin C/
        Vitamin D/
        Vitamin E/
        Retina Blood Vessel/   <-- automatically excluded by this script

Usage
-----
    pip install -r requirements.txt
    python train_efficientnetv2s.py --data_dir /path/to/dataset --output_dir ./runs/effnetv2s

What this script does
----------------------
1. Scans data_dir, drops any folder that looks like "retina" (case-insensitive), so you
   don't have to physically delete the Retina Blood Vessel folder.
2. Builds a stratified train/val/test split (default 70/15/15) using sklearn, so class
   balance is preserved -- important given the vitamin classes are unevenly sized.
3. Applies the augmentation you specified: RandomResizedCrop-style 256->224 crop,
   horizontal flip, rotation (+/-15deg), color jitter, and random affine translation
   for the TRAIN split only. Val/test just get resize + ImageNet normalization.
4. Loads EfficientNetV2-S with ImageNet weights and replaces the classifier head with
   a fresh Linear layer for your 5 vitamin classes.
5. Trains in two phases:
     Phase 1 - backbone frozen, train only the new classifier head (fast warmup).
     Phase 2 - unfreeze the whole network, fine-tune end-to-end at a lower LR.
   Model selection is done on validation macro-F1 (better than accuracy for
   imbalanced/small classes), not just val loss.
6. Evaluates on the held-out test split: overall accuracy, per-class precision/
   recall/F1 (classification_report), and a confusion matrix plotted to PNG.
7. Saves the best checkpoint plus a JSON of final metrics.

Grad-CAM (for the explainability step in the paper) is in the companion script
`gradcam_efficientnetv2s.py` -- run it after training, pointing it at the saved
checkpoint.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupt files instead of crashing

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_EXTS = {".png", ".jpg", ".jpeg"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset indexing (with retina-class exclusion + mask-folder exclusion)
# ---------------------------------------------------------------------------

def is_excluded_folder(folder_name: str) -> bool:
    name = folder_name.lower()
    return "retina" in name or "mask" in name


def index_dataset(data_dir: Path):
    """Walk data_dir/<class_name>/*.ext and return (paths, labels, class_names)."""
    class_dirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir() and not is_excluded_folder(d.name)]
    )
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {data_dir} (after excluding retina/mask folders).")

    class_names = [d.name for d in class_dirs]
    paths, labels = [], []
    for label_idx, class_dir in enumerate(class_dirs):
        files = [
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VALID_EXTS and not is_excluded_folder(p.parent.name)
        ]
        if not files:
            print(f"[warn] no images found in {class_dir}")
        paths.extend(files)
        labels.extend([label_idx] * len(files))

    print(f"Found {len(paths)} images across {len(class_names)} classes: {class_names}")
    for i, cname in enumerate(class_names):
        print(f"  {cname}: {labels.count(i)} images")
    return paths, labels, class_names


def stratified_split(paths, labels, val_size=0.15, test_size=0.15, seed=42):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(val_size + test_size), stratify=labels, random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=relative_test, stratify=temp_labels, random_state=seed
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


# ---------------------------------------------------------------------------
# Dataset / transforms
# ---------------------------------------------------------------------------

class VitaminDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[warn] failed to load {path} ({e}); using a gray placeholder image")
            image = Image.new("RGB", (256, 256), color=(128, 128, 128))
        image = self.transform(image)
        return image, label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = trainable


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_labels, all_preds


def train_phase(model, train_loader, val_loader, device, epochs, lr, phase_name, best_f1, ckpt_path):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"[{phase_name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best model saved (val_f1={best_f1:.4f})")

    return best_f1


def evaluate_test_set(model, test_loader, device, class_names, output_dir: Path):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print("\nTest set classification report:\n", report)
    print("Confusion matrix (rows=true, cols=predicted):\n", cm)

    (output_dir / "classification_report.txt").write_text(report)
    np.savetxt(output_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("EfficientNetV2-S - Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
        plt.close()
        print(f"Saved confusion matrix plot to {output_dir / 'confusion_matrix.png'}")
    except ImportError:
        print("[info] matplotlib/seaborn not installed - skipped confusion matrix plot (CSV still saved).")

    return report, cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S on the vitamin deficiency dataset.")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the extracted Kaggle dataset root (contains one folder per class).")
    parser.add_argument("--output_dir", type=str, default="./runs/effnetv2s")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Phase 1: head-only training epochs.")
    parser.add_argument("--finetune_epochs", type=int, default=20, help="Phase 2: full fine-tuning epochs.")
    parser.add_argument("--warmup_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Index dataset (retina/mask folders auto-excluded)
    data_dir = Path(args.data_dir)
    paths, labels, class_names = index_dataset(data_dir)

    # 2. Stratified split
    (train_p, train_l), (val_p, val_l), (test_p, test_l) = stratified_split(
        paths, labels, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    print(f"Split sizes -> train: {len(train_p)}, val: {len(val_p)}, test: {len(test_p)}")

    # Save the split for reproducibility / reporting in the paper
    split_record = {
        "class_names": class_names,
        "train": [str(p) for p in train_p],
        "val": [str(p) for p in val_p],
        "test": [str(p) for p in test_p],
    }
    (output_dir / "data_split.json").write_text(json.dumps(split_record, indent=2))

    # 3. Transforms + loaders
    train_tf, eval_tf = build_transforms()
    train_ds = VitaminDataset(train_p, train_l, train_tf)
    val_ds = VitaminDataset(val_p, val_l, eval_tf)
    test_ds = VitaminDataset(test_p, test_l, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Model
    model = build_model(num_classes=len(class_names)).to(device)
    ckpt_path = output_dir / "best_model.pt"
    best_f1 = 0.0

    # 5. Phase 1 - warm up the new classifier head only
    print("\n=== Phase 1: training classifier head (backbone frozen) ===")
    set_backbone_trainable(model, trainable=False)
    best_f1 = train_phase(model, train_loader, val_loader, device,
                           epochs=args.warmup_epochs, lr=args.warmup_lr,
                           phase_name="warmup", best_f1=best_f1, ckpt_path=ckpt_path)

    # 6. Phase 2 - fine-tune the whole network at a lower LR
    print("\n=== Phase 2: fine-tuning full network ===")
    set_backbone_trainable(model, trainable=True)
    best_f1 = train_phase(model, train_loader, val_loader, device,
                           epochs=args.finetune_epochs, lr=args.finetune_lr,
                           phase_name="finetune", best_f1=best_f1, ckpt_path=ckpt_path)

    # 7. Load best checkpoint and evaluate on held-out test set
    print(f"\nLoading best checkpoint (val_f1={best_f1:.4f}) for final test evaluation...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    report, cm = evaluate_test_set(model, test_loader, device, class_names, output_dir)

    metrics_summary = {
        "best_val_macro_f1": best_f1,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))
    print(f"\nDone. Best model + metrics saved under: {output_dir}")


if __name__ == "__main__":
    main()

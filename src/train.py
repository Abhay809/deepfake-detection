"""
Training script for DeepFake detection model.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm

from .dataset import DeepFakeDataset, collate_with_aux
from .models import DeepFakeClassifier


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data-root", default=None, help="Override data root; real/fake under it")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    preprocess_cfg = config.get("preprocessing", {})
    seed = config.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)

    real_dir = data_cfg["real_dir"]
    fake_dir = data_cfg["fake_dir"]
    if args.data_root:
        real_dir = os.path.join(args.data_root, "real")
        fake_dir = os.path.join(args.data_root, "fake")

    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)
    use_freq = preprocess_cfg.get("use_frequency_features", True)
    use_tex = preprocess_cfg.get("use_texture_features", True)
    augment = preprocess_cfg.get("augment", True)

    full_dataset = DeepFakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        image_size=image_size,
        is_train=False,
        augment=False,
        use_frequency=use_freq,
        use_texture=use_tex,
    )
    n = len(full_dataset)
    tr = int(n * data_cfg["train_ratio"])
    val = int(n * data_cfg["val_ratio"])
    te = n - tr - val
    train_sub, val_sub, _ = random_split(full_dataset, [tr, val, te])
    train_indices, val_indices = train_sub.indices, val_sub.indices

    train_dataset = DeepFakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        image_size=image_size,
        is_train=True,
        augment=augment,
        use_frequency=use_freq,
        use_texture=use_tex,
    )
    val_dataset = DeepFakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        image_size=image_size,
        is_train=False,
        use_frequency=use_freq,
        use_texture=use_tex,
    )
    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_with_aux,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_aux,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFakeClassifier(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        use_aux=use_freq or use_tex,
        aux_channels=2 if (use_freq and use_tex) else 1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler_name = train_cfg.get("lr_scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    freeze_epochs = model_cfg.get("freeze_backbone_epochs", 0)
    best_val_acc = 0.0
    patience = train_cfg.get("early_stopping_patience", 10)
    patience_counter = 0

    for epoch in range(train_cfg["epochs"]):
        if epoch == freeze_epochs:
            model.unfreeze_backbone()
            print("Backbone unfrozen.")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")
        for rgb, aux, labels in pbar:
            rgb, labels = rgb.to(device), labels.to(device)
            aux = aux.to(device) if aux is not None else None
            optimizer.zero_grad()
            logits = model(rgb, aux)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        train_acc = correct / total
        if scheduler_name == "cosine" or scheduler_name == "step":
            scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for rgb, aux, labels in val_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                aux = aux.to(device) if aux is not None else None
                logits = model(rgb, aux)
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total else 0.0
        if scheduler_name == "plateau":
            scheduler.step(val_acc)

        print(f"Epoch {epoch+1} — Train loss: {running_loss/len(train_loader):.4f} Train acc: {train_acc:.4f} Val acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            path = os.path.join(train_cfg["checkpoint_dir"], "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "config": config,
            }, path)
            print(f"  Saved best model (val acc {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Training done. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

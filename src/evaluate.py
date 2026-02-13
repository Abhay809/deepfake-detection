"""
Evaluate the trained DeepFake detection model on test set.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import DeepFakeDataset, collate_with_aux
from .models import DeepFakeClassifier


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    preprocess_cfg = config.get("preprocessing", {})

    real_dir = data_cfg["real_dir"]
    fake_dir = data_cfg["fake_dir"]
    if args.data_root:
        real_dir = os.path.join(args.data_root, "real")
        fake_dir = os.path.join(args.data_root, "fake")

    use_freq = preprocess_cfg.get("use_frequency_features", True)
    use_tex = preprocess_cfg.get("use_texture_features", True)

    dataset = DeepFakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        image_size=data_cfg["image_size"],
        is_train=False,
        use_frequency=use_freq,
        use_texture=use_tex,
    )
    # Use full dataset as "test" for evaluation; in practice you'd use held-out test split
    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collate_with_aux,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_config = ckpt.get("config", config)
    m_cfg = saved_config.get("model", model_cfg)
    pre_cfg = saved_config.get("preprocessing", preprocess_cfg)
    use_freq = pre_cfg.get("use_frequency_features", True)
    use_tex = pre_cfg.get("use_texture_features", True)

    model = DeepFakeClassifier(
        backbone_name=m_cfg["backbone"],
        pretrained=False,
        num_classes=m_cfg["num_classes"],
        dropout=m_cfg["dropout"],
        use_aux=use_freq or use_tex,
        aux_channels=2 if (use_freq and use_tex) else 1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for rgb, aux, labels in loader:
            rgb, labels = rgb.to(device), labels.to(device)
            aux = aux.to(device) if aux is not None else None
            logits = model(rgb, aux)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    print(f"Overall accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

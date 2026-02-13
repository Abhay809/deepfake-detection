"""
Single-image and video-frame inference for DeepFake detection.
"""

import os
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image

from .preprocessing import (
    get_val_transforms,
    compute_frequency_features,
    compute_texture_features,
)
from .models import DeepFakeClassifier


def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config_path: str = "config.yaml", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_config = ckpt.get("config", config)
    m_cfg = saved_config.get("model", config["model"])
    pre_cfg = saved_config.get("preprocessing", config.get("preprocessing", {}))
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
    return model, config, device, (use_freq, use_tex)


def predict_image(
    image_path: str,
    model,
    config,
    device,
    use_aux_flags,
    transform=None,
):
    """Run prediction on a single image. Returns label (0=real, 1=fake) and probability."""
    image_size = config["data"]["image_size"]
    if transform is None:
        transform = get_val_transforms(image_size)

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    rgb = transform(img_pil).unsqueeze(0).to(device)

    use_freq, use_tex = use_aux_flags
    aux_list = []
    if use_freq:
        aux_list.append(compute_frequency_features(img_np))
    if use_tex:
        aux_list.append(compute_texture_features(img_np))
    aux = torch.from_numpy(np.stack(aux_list, axis=0)).float().unsqueeze(0).to(device) if aux_list else None

    with torch.no_grad():
        logits = model(rgb, aux)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1).item()
        prob_fake = probs[0, 1].item()
    return pred, prob_fake


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    model, config, device, use_aux = load_model(args.checkpoint, args.config)
    pred, prob_fake = predict_image(args.image, model, config, device, use_aux)
    label = "Fake" if pred == 1 else "Real"
    print(f"Prediction: {label} (P(fake) = {prob_fake:.4f})")


if __name__ == "__main__":
    main()

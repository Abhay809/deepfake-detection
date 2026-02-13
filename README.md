# DeepFake Detection and Prevention – A Comprehensive AI Approach

An intelligent, high-accuracy system to identify manipulated or AI-generated media by combining **CNNs**, **transfer learning**, and **artifact-aware preprocessing** (frequency-domain and texture cues). Target: **up to ~95% accuracy** on real vs deepfake classification.

## Features

- **Transfer learning** with EfficientNet-B0 / ResNet50 / MobileNet-V3
- **Artifact analysis**: frequency-domain (FFT) and texture-consistency features to capture blending errors and distortions
- **Robust preprocessing**: normalized RGB + optional auxiliary channels for the model
- **Training pipeline**: train/val split, augmentation, checkpointing, early stopping
- **Deployment**: Streamlit web app for image and video-frame upload and prediction

## Project structure

```
d1/
├── config.yaml          # Hyperparameters and paths
├── requirements.txt     # Python dependencies
├── run_train.py         # Train model
├── run_evaluate.py      # Evaluate on dataset
├── run_predict.py       # CLI prediction for one image
├── app.py               # Streamlit deployment app
├── data/
│   ├── real/            # Authentic images
│   └── fake/            # Deepfake / manipulated images
├── checkpoints/         # Saved best model (after training)
├── src/
│   ├── preprocessing.py # FFT, texture, transforms
│   ├── dataset.py       # PyTorch dataset
│   ├── models.py        # CNN + transfer learning
│   ├── train.py         # Training loop
│   ├── evaluate.py      # Metrics and report
│   └── predict.py       # Inference
└── scripts/
    └── download_sample_data.py  # Data folder setup hints
```

## Setup

1. **Create environment and install dependencies**

   ```bash
   cd c:\Users\lenovo\Desktop\d1
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare data**

   - Put **authentic** images in `data/real/`
   - Put **deepfake / manipulated** images in `data/fake/`
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

   For quick testing, use at least ~20–50 images per class. For better accuracy (e.g. ~95%), use larger datasets such as:

   - [Face Forensics++](https://github.com/ondyari/FaceForensics)
   - [Celeb-DF](https://github.com/yuezunliu-ceb/Celeb-DF)
   - [DFDC on Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge)

   Optional: run `python scripts/download_sample_data.py` to see folder structure and dataset links.

## Run

### 1. Train

```bash
python run_train.py
```

Optional: use another config or data root:

```bash
python run_train.py --config config.yaml --data-root path/to/data
```

Training saves the best model to `checkpoints/best.pt` (by validation accuracy).

### 2. Evaluate

```bash
python run_evaluate.py --checkpoint checkpoints/best.pt
```

Prints classification report, confusion matrix, and overall accuracy.

### 3. Predict on a single image (CLI)

```bash
python run_predict.py path/to/image.jpg
```

### 4. Deploy – Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown (e.g. `http://localhost:8501`). You can:

- Upload an **image** or **video** (a middle frame is analyzed for video)
- Get a **Real / Fake** prediction and confidence

## Configuration

Edit `config.yaml` to change:

- **data**: `image_size`, `batch_size`, `real_dir`, `fake_dir`
- **model**: `backbone` (`efficientnet_b0`, `resnet50`, `mobilenet_v3_small`), `dropout`, `freeze_backbone_epochs`
- **training**: `epochs`, `learning_rate`, `early_stopping_patience`, `checkpoint_dir`
- **preprocessing**: `use_frequency_features`, `use_texture_features`, `augment`

## Summary

| Step    | Command / action |
|--------|-------------------|
| Install | `pip install -r requirements.txt` |
| Data    | Add images to `data/real/` and `data/fake/` |
| Train   | `python run_train.py` |
| Evaluate| `python run_evaluate.py` |
| Predict | `python run_predict.py <image>` |
| Deploy  | `streamlit run app.py` |

With a sufficiently large and balanced dataset, this pipeline can reach **up to ~95% accuracy** on real vs deepfake classification by leveraging transfer learning and artifact-aware features.

---

## Deploy (GitHub + Streamlit Cloud)

1. **Push the project to GitHub**
   - Create a new repo on [github.com](https://github.com), then in your project folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```
   Use a [Personal Access Token](https://github.com/settings/tokens) as password when prompted.

2. **Deploy on Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
   - **New app** → select your repo, branch `main`, main file path **`app.py`** → **Deploy**.

3. **Model on the cloud**
   - Either commit `checkpoints/best.pt` after training locally and push, or upload `best.pt` to a URL and set **Secrets** in the Streamlit Cloud app to: `CHECKPOINT_URL = "https://..."`.

See **[DEPLOY.md](DEPLOY.md)** for the full step-by-step guide.

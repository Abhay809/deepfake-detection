# Deploy DeepFake Detection with GitHub + Streamlit Cloud

Follow these steps to put your app on GitHub and deploy it on **Streamlit Community Cloud** (free).

---

## 1. Create a GitHub repository

1. Go to [github.com](https://github.com) and sign in.
2. Click **New repository** (or **+** → **New repository**).
3. Set:
   - **Repository name:** e.g. `deepfake-detection`
   - **Visibility:** Public (required for free Streamlit Cloud).
   - Leave “Add a README” unchecked if you’re pushing an existing project.
4. Click **Create repository**.

---

## 2. Push your project from your PC

Open a terminal in your project folder (e.g. `c:\Users\lenovo\Desktop\d1`) and run:

```bash
# Go to project folder
cd c:\Users\lenovo\Desktop\d1

# Initialize Git (if this folder is not already a repo)
git init

# Add all files (checkpoints/*.pt and venv are ignored via .gitignore)
git add .
git commit -m "Initial commit: DeepFake Detection app"

# Add your GitHub repo as remote (replace YOUR_USERNAME and REPO_NAME with yours)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push (use main or master depending on your default branch)
git branch -M main
git push -u origin main
```

When Git asks for credentials:

- **Username:** your GitHub username  
- **Password:** use a **Personal Access Token** (not your GitHub password).  
  Create one: GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Generate new token**, enable `repo`, then paste the token as the password.

---

## 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Choose:
   - **Repository:** `YOUR_USERNAME/REPO_NAME`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.

The first run may take a few minutes (installing dependencies and loading the model).

---

## 4. Model checkpoint for the deployed app

The app needs a trained model file: `checkpoints/best.pt`.

**Option A – Commit the checkpoint (simplest)**  
If the file is not too large for your repo (~50–100 MB can be acceptable):

1. Train locally: `python run_train.py`
2. Add and push the checkpoint (use `-f` because `checkpoints/*.pt` is in `.gitignore`):
   ```bash
   git add -f checkpoints/best.pt
   git commit -m "Add trained model checkpoint"
   git push
   ```
3. Redeploy or wait for Streamlit Cloud to redeploy from the new commit.

**Option B – Host checkpoint elsewhere (no large file in GitHub)**  
Use a URL so the app downloads the model at startup:

1. Upload `checkpoints/best.pt` somewhere public (e.g. GitHub Releases, Google Drive direct link, or another file host).
2. In Streamlit Cloud: open your app → **Settings** (⋮) → **Secrets**.
3. Add:
   ```toml
   CHECKPOINT_URL = "https://your-url-to-best.pt"
   ```
4. Save and let the app restart. The app will download the checkpoint on first load.

---

## 5. Useful commands after deployment

| Task              | Command / action                          |
|-------------------|-------------------------------------------|
| Update app code   | Edit locally → `git add .` → `git commit -m "..."` → `git push` |
| Update checkpoint | Re-train, then Option A or B above        |
| View logs         | Streamlit Cloud → your app → **Manage app** → **Logs** |

---

## 6. Optional: GitHub CLI

If you use [GitHub CLI](https://cli.github.com/):

```bash
cd c:\Users\lenovo\Desktop\d1
git init
git add .
git commit -m "Initial commit: DeepFake Detection app"
gh repo create deepfake-detection --public --source=. --push
```

Then deploy the repo on [share.streamlit.io](https://share.streamlit.io) as in step 3.

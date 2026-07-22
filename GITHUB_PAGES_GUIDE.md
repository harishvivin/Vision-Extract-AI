# 🚀 Hosting Vision Extract AI on GitHub Pages

This project is fully configured to run as a **pure static web application on GitHub Pages**. 
Visitors can view all **10 extracted page outputs**, compare bounding box overlays vs cropped object PNGs, inspect AI model confidence scores & prompts, view telemetry logs, and download individual PNGs or the full `all_extracted_objects.zip` package — **without needing to run Python, FastAPI, or backend servers!**

---

## 📌 Quick Setup Guide (Push to your GitHub Account)

### Step 1: Create a New GitHub Repository
1. Go to [GitHub - New Repository](https://github.com/new).
2. Set Repository Name: `Vision-Extract-AI` (or any name you choose).
3. Set Visibility: **Public**.
4. Leave "Add a README file" **unchecked** (since we already have local git files ready).
5. Click **Create repository**.

---

### Step 2: Push your Local Code to GitHub
Run the following commands in your terminal inside your project folder (`c:\Users\haris\Desktop\Vision Extract AI`):

```bash
# Add your GitHub repository as remote (replace YOUR_USERNAME with your GitHub handle)
git remote add origin https://github.com/YOUR_USERNAME/Vision-Extract-AI.git

# Set default branch to main
git branch -M main

# Push code to GitHub
git push -u origin main
```

---

### Step 3: Enable GitHub Pages

You have **two easy ways** to enable GitHub Pages:

#### Method A: Using GitHub Actions (Automated Deployment - Recommended)
1. Go to your GitHub repository -> **Settings** -> **Pages**.
2. Under **Build and deployment** -> **Source**, select **GitHub Actions**.
3. That's it! GitHub will automatically trigger the `.github/workflows/deploy.yml` workflow and publish your site at:
   `https://YOUR_USERNAME.github.io/Vision-Extract-AI/`

#### Method B: Deploying from `/docs` Folder (Instant 1-Click)
1. Go to your GitHub repository -> **Settings** -> **Pages**.
2. Under **Build and deployment** -> **Source**, select **Deploy from a branch**.
3. Under **Branch**, select `main` branch and `/docs` folder.
4. Click **Save**.
5. Your page will be live in ~1-2 minutes at:
   `https://YOUR_USERNAME.github.io/Vision-Extract-AI/`

---

## 🌟 Static Hosting Features Included

- **No Backend Server Required**: Pre-compiled static dataset (`data/results.json` & `data/logs.json`) delivers instant loading.
- **Interactive UI**: View, search, toggle bounding box overlay vs SAM 2 segmentations across all 10 pages.
- **Full Downloads**: Download individual cropped PNGs or the complete `all_extracted_objects.zip`.
- **Hybrid Ready**: If hosted locally with `python app.py` and `npm run dev`, it connects dynamically to the live AI backend pipeline.

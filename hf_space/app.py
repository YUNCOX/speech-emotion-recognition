import os
import gdown

# 1. Download the 500MB model BEFORE importing the demo to prevent crashes
os.makedirs("models", exist_ok=True)
model_path = "models/best_ser_final.pkl"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    url = 'https://drive.google.com/uc?id=1yXk90_J8wll8gQL5onUOg78Iso5HTIRU'
    gdown.download(url, model_path, quiet=False, fuzzy=True)

# 2. Import and run your existing Gradio demo
from app.demo import demo

if __name__ == "__main__":
    demo.launch()
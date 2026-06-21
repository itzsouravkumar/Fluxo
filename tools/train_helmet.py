#!/usr/bin/env python3
"""
FLUXO Helmet Classifier Training Pipeline
Uses the Kaggle dataset 'abuzarkhaaan/helmet-dataset-cls'.
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# Configuration
KAGGLE_DS_PATH = Path("/Users/itz_sour4v/.cache/kagglehub/datasets/abuzarkhaaan/helmet-dataset-cls/versions/3/New folder (3)")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS = 10

def train_helmet():
    print(f"\n{'='*50}\nTraining Helmet Classifier\n{'='*50}")
    
    if not KAGGLE_DS_PATH.exists():
        print(f"Error: Dataset directory {KAGGLE_DS_PATH} missing.")
        return False
        
    try:
        model = YOLO("yolov8n-cls.pt")  # Nano classification model
        
        # Train the model
        results = model.train(
            data=str(KAGGLE_DS_PATH.absolute()),
            epochs=EPOCHS,
            imgsz=224,
            batch=32,
            device="mps" if os.name == 'posix' else "cpu",
            project=str(Path("datasets") / "runs"),
            name="helmet_kaggle",
            exist_ok=True,
            verbose=False
        )
        
        # Move the best model to our models directory
        best_model_path = Path("datasets") / "runs" / "helmet_kaggle" / "weights" / "best.pt"
        if best_model_path.exists():
            target_path = MODELS_DIR / "fluxo_helmet_v1.pt"
            shutil.copy(str(best_model_path), str(target_path))
            print(f"\nSuccess! Saved trained model to {target_path}")
            return True
        else:
            print("Error: Training completed but best.pt not found.")
            return False
            
    except Exception as e:
        print(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    train_helmet()

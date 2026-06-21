#!/usr/bin/env python3
"""
FLUXO Classifier Training Pipeline
Scrapes images using bing-image-downloader and trains YOLOv8 classification models.
"""

import os
import shutil
from pathlib import Path
from bing_image_downloader import downloader
from ultralytics import YOLO

# Configuration
DATASETS_DIR = Path("datasets")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_PER_CLASS = 150
EPOCHS = 8

# Define the tasks, classes, and search queries
TASKS = {
    "helmet": {
        "classes": {
            "helmet": [
                "motorcycle rider wearing helmet close up",
                "person wearing motorcycle helmet",
            ],
            "no_helmet": [
                "motorcycle rider without helmet close up",
                "person riding motorcycle bare head",
                "indian motorcycle rider without helmet",
            ],
        },
        "output_model": "fluxo_helmet_v1.pt",
    },
    "seatbelt": {
        "classes": {
            "with_seatbelt": [
                "car driver wearing seatbelt close up",
                "person wearing seat belt in car",
            ],
            "without_seatbelt": [
                "car driver not wearing seatbelt close up",
                "person driving car no seatbelt",
            ],
        },
        "output_model": "fluxo_seatbelt_v1.pt",
    },
    "mobile_phone": {
        "classes": {
            "using_phone": [
                "driver talking on mobile phone in car",
                "person using phone while driving",
            ],
            "no_phone": [
                "driver holding steering wheel close up",
                "person driving looking forward",
            ],
        },
        "output_model": "fluxo_mobile_phone_v1.pt",
    },
}

def clean_dataset_dir(task_name):
    task_dir = DATASETS_DIR / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir

def scrape_data(task_name, classes):
    print(f"\n{'='*50}\nScraping data for task: {task_name}\n{'='*50}")
    task_dir = clean_dataset_dir(task_name)
    train_dir = task_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name, queries in classes.items():
        class_dir = train_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Divide the IMAGES_PER_CLASS by the number of queries
        limit_per_query = max(1, IMAGES_PER_CLASS // len(queries))
        
        for query in queries:
            print(f"--> Scraping '{query}' for class '{class_name}'")
            try:
                downloader.download(
                    query,
                    limit=limit_per_query,
                    output_dir=str(class_dir.parent),
                    adult_filter_off=False,
                    force_replace=False,
                    timeout=10,
                    verbose=False
                )
                # bing-image-downloader creates a folder named after the query
                query_folder = class_dir.parent / query
                if query_folder.exists():
                    # Move all files from query_folder to class_dir
                    for f in query_folder.iterdir():
                        if f.is_file():
                            shutil.move(str(f), str(class_dir / f.name))
                    shutil.rmtree(query_folder)
            except Exception as e:
                print(f"Error scraping query '{query}': {e}")
                
def train_model(task_name, output_model):
    print(f"\n{'='*50}\nTraining model for task: {task_name}\n{'='*50}")
    task_dir = DATASETS_DIR / task_name
    
    # Check if dataset exists and has images
    train_dir = task_dir / "train"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        print(f"Error: Dataset directory {train_dir} is missing or empty.")
        return False
        
    try:
        model = YOLO("yolov8n-cls.pt")  # Use nano classification model
        
        # Train the model
        results = model.train(
            data=str(task_dir.absolute()),
            epochs=EPOCHS,
            imgsz=224,
            batch=16,
            device="mps" if os.name == 'posix' else "cpu",  # Use MPS on Mac if available
            project=str(DATASETS_DIR / "runs"),
            name=task_name,
            exist_ok=True,
            verbose=False
        )
        
        # Move the best model to our models directory
        best_model_path = DATASETS_DIR / "runs" / task_name / "weights" / "best.pt"
        if best_model_path.exists():
            target_path = MODELS_DIR / output_model
            shutil.copy(str(best_model_path), str(target_path))
            print(f"\nSuccess! Saved trained model to {target_path}")
            return True
        else:
            print("Error: Training completed but best.pt not found.")
            return False
            
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def main():
    for task_name, config in TASKS.items():
        scrape_data(task_name, config["classes"])
        train_model(task_name, config["output_model"])
        
    print("\nAll training tasks completed!")

if __name__ == "__main__":
    main()

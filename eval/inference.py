"""eval/inference.py"""

import sys
import os
# 경로 설정 문제 해결
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_image_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]

def predict_images(model, processor, image_paths, device):
    probs = []

    with torch.no_grad():
        for path in tqdm(image_paths, desc="Inferencing"):
            try:
                image = Image.open(path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)

                prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
                fake_prob = prob[0][1].item()
                probs.append(fake_prob)
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return probs

def evaluate_detection(real_probs, fake_probs, best_threshold=0.5):
    y_true = [0] * len(real_probs) + [1] * len(fake_probs)
    all_probs = real_probs + fake_probs
    y_pred = [1 if p >= best_threshold else 0 for p in all_probs]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"--- Detection Results (Threshold: {best_threshold}) ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f} (Fake 기준)")
    print(f"Recall    : {rec:.4f} (Fake 기준)")
    print(f"F1-score  : {f1:.4f} (Fake 기준)")

    return acc, prec, rec, f1

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = "google/vit-base-patch16-224"
    BEST_THRESHOLD = 0.5

    # 🚀 실제 평가용 세팅 복구 (경로 분리)
    REAL_DIR = os.path.join(BASE_DIR, 'data', 'img_align_celeba')
    FAKE_DIR = os.path.join(BASE_DIR, 'data', 'fake_images')

    os.makedirs(FAKE_DIR, exist_ok=True)

    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device).eval()

    real_paths = get_image_paths(REAL_DIR)
    fake_paths = get_image_paths(FAKE_DIR)

    print(f"Processing Real Images ({len(real_paths)}장)...")
    real_probs = predict_images(model, processor, real_paths, device)

    print(f"Processing Fake Images ({len(fake_paths)}장)...")
    fake_probs = predict_images(model, processor, fake_paths, device)

    evaluate_detection(real_probs, fake_probs, best_threshold=BEST_THRESHOLD)
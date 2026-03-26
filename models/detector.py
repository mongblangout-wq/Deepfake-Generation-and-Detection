"""models/detector.py"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification

class DeepfakeDetector(nn.Module):
    def __init__(self, model_name_or_path="google/vit-base-patch16-224", num_labels=2):
        super().__init__()
        # 사전 학습된 ViT 모델 불러오기 및 라벨 수 설정
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True # 라벨 수가 다를 경우 분류기 층 초기화
        )

    def forward(self, pixel_values):
        # transformers 모델의 출력에서 logits만 반환
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
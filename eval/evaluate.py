"""eval/evaluate.py"""

import os
import sys

# 🚨 OpenMP 중복 로드 에러(OMP: Error #15) 해결을 위한 환경 변수 강제 허용
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 경로 설정 문제 해결
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

import torch
import lpips
import numpy as np
from PIL import Image
from pytorch_fid import fid_score
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

def evaluate_lpips(real_folder, fake_folder, device, sample_size=20):
    # LPIPS 모델 로드 (AlexNet 기반)
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    lpips_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    real_ds = ImageDataset(real_folder, transform=lpips_tf)
    fake_ds = ImageDataset(fake_folder, transform=lpips_tf)

    n_samples = min(len(real_ds), len(fake_ds), sample_size)
    real_loader = DataLoader(Subset(real_ds, list(range(n_samples))), batch_size=1, shuffle=False)
    fake_loader = DataLoader(Subset(fake_ds, list(range(n_samples))), batch_size=1, shuffle=False)

    total_lpips = []
    with torch.no_grad():
        for (r_img, _), (f_img, _) in zip(real_loader, fake_loader):
            r_img, f_img = r_img.to(device), f_img.to(device)
            dist = lpips_fn(f_img, r_img)
            total_lpips.append(dist.item())

    lpips_scores = np.array(total_lpips)
    print(f"[LPIPS] Evaluated {n_samples} pairs")
    print(f"Mean: {lpips_scores.mean():.4f} | Median: {np.median(lpips_scores):.4f} | Std: {lpips_scores.std():.4f}")

def evaluate_fid(real_folder, fake_folder, device):
    print("Calculating FID Score...")
    # 테스트용 데이터가 20장뿐이므로 batch_size를 4로 줄임
    fid_value = fid_score.calculate_fid_given_paths(
        [real_folder, fake_folder],
        batch_size=4,
        device=device,
        dims=2048
    )
    print(f"[FID] Score: {fid_value:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 🚀 실제 평가용 세팅 복구 (경로 분리)
    REAL_DIR = os.path.join(BASE_DIR, 'data', 'img_align_celeba')
    FAKE_DIR = os.path.join(BASE_DIR, 'data', 'fake_images') # 나중에 생성된 가짜 이미지가 들어갈 폴더

    # Fake 폴더가 없으면 에러가 나지 않도록 생성해둠
    os.makedirs(FAKE_DIR, exist_ok=True)

    evaluate_lpips(REAL_DIR, FAKE_DIR, device)
    evaluate_fid(REAL_DIR, FAKE_DIR, device)
"""data/dataset.py"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ProGAN 등 기본 이미지 전처리 Transform 정의
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])


class CelebADataset(Dataset):
    def __init__(self, img_dir, img_num_list=None, transform=None):
        """
        img_dir: 이미지가 저장된 폴더 경로
        img_num_list: 불러올 이미지 번호 리스트 (DDPM 등에서 데이터 분할 시 사용)
        transform: 이미지 전처리
        """
        self.transform = transform

        if img_num_list is not None:
            # DDPM: 리스트가 주어지면 (예: 1 -> 000001.jpg) 해당 파일만 매핑
            self.img_paths = [os.path.join(img_dir, f"{int(num):06d}.jpg") for num in img_num_list]
        else:
            # ProGAN: 리스트가 없으면 폴더 내 전체 .jpg 파일 로드
            self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 1  # 1은 Real image의 라벨(ProGAN Discriminator 등에서 사용)
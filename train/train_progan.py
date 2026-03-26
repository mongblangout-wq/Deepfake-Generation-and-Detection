"""train/train_progan.py"""

import sys
import os
# 1. 파이썬이 상위 폴더(프로젝트 최상단)를 인식할 수 있도록 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

# 2. 파일명을 progan.py로 저장했으므로 수정 (models.model -> models.progan)
from models.progan import Generator, Discriminator
from data.dataset import CelebADataset, transform

class Trainer():
    def __init__(self, steps, device, train_loader, val_loader, checkpoint_path=None):
        self.steps = steps
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 모델 초기화
        self.generator = Generator(steps).to(device)
        self.discriminator = Discriminator(steps).to(device)

        # Loss 및 Optimizer
        self.criterion = nn.BCELoss()
        self.g_optimizer = Adam(self.generator.parameters(), lr=0.001, betas=(0.0, 0.99))
        self.d_optimizer = Adam(self.discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))

        self.alpha = 0.0
        self.history = {'g_train': [], 'd_train': [], 'g_val': [], 'd_val': []}
        self.start_epoch = 0

        # Checkpoint 불러오기
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.generator.load_state_dict(checkpoint['g_state_dict'])
            self.discriminator.load_state_dict(checkpoint['d_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.history = checkpoint['history']
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {self.start_epoch - 1}")

    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for batch_idx, (real, _) in enumerate(self.train_loader):
            real = real.to(self.device)
            bs = real.size(0)

            # Label 설정
            real_lbl = torch.ones(bs, 1, device=self.device)
            fake_lbl = torch.zeros(bs, 1, device=self.device)

            # 1. Discriminator 학습
            self.d_optimizer.zero_grad()
            z = torch.randn(bs, 128, 1, 1, device=self.device)
            fake = self.generator(z)

            d_real = self.discriminator(real, self.alpha)
            d_fake = self.discriminator(fake.detach(), self.alpha)

            d_loss = self.criterion(d_fake, fake_lbl) + self.criterion(d_real, real_lbl)
            d_loss.backward()
            self.d_optimizer.step()
            d_loss_avg += d_loss.item() / 2

            # 2. Generator 학습
            self.g_optimizer.zero_grad()
            fake_pred = self.discriminator(fake, self.alpha)
            g_loss = self.criterion(fake_pred, real_lbl)
            g_loss.backward()
            self.g_optimizer.step()
            g_loss_avg += g_loss.item()

            # Alpha 업데이트 (Fade-in)
            self.alpha = min(1.0, self.alpha + 1 / (len(self.train_loader) * 5))

        return g_loss_avg / len(self.train_loader), d_loss_avg / len(self.train_loader)

    def valid_epoch(self):
        self.generator.eval()
        self.discriminator.eval()
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        with torch.no_grad():
            for real, _ in self.val_loader:
                real = real.to(self.device)
                bs = real.size(0)
                real_lbl = torch.ones(bs, 1, device=self.device)
                fake_lbl = torch.zeros(bs, 1, device=self.device)

                z = torch.randn(bs, 128, 1, 1, device=self.device)
                fake = self.generator(z)

                d_real = self.discriminator(real, self.alpha)
                d_fake = self.discriminator(fake, self.alpha)

                d_loss_avg += (self.criterion(d_fake, fake_lbl) + self.criterion(d_real, real_lbl)).item() / 2
                g_loss_avg += self.criterion(self.discriminator(fake, self.alpha), real_lbl).item()

        return g_loss_avg / len(self.val_loader), d_loss_avg / len(self.val_loader)

    def run(self, epochs):
        for epoch in range(self.start_epoch, epochs):
            g_t, d_t = self.train_epoch()
            g_v, d_v = self.valid_epoch()

            self.history['g_train'].append(g_t)
            self.history['d_train'].append(d_t)
            self.history['g_val'].append(g_v)
            self.history['d_val'].append(d_v)

            print(f"Epoch {epoch}: G_loss {g_t:.4f}/{g_v:.4f} | D_loss {d_t:.4f}/{d_v:.4f}")

            # 모델 저장
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'g_state_dict': self.generator.state_dict(),
                    'd_state_dict': self.discriminator.state_dict(),
                    'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'history': self.history
                }, f"checkpoint_epoch_{epoch}.pth")

        return self.history


if __name__ == "__main__":
    # 데이터 경로 절대 경로로 확실하게 지정
    BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    data_dir = os.path.join(BASE_DIR, 'data', 'img_align_celeba')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Data Directory: {data_dir}")

    # 데이터 로더 준비
    full_dataset = CelebADataset(data_dir, transform=transform)
    print(f"Total images loaded: {len(full_dataset)}")

    # 예외 처리: 이미지를 찾지 못했을 경우 명확한 에러 발생
    if len(full_dataset) == 0:
        raise ValueError("이미지를 한 장도 불러오지 못했습니다. 폴더 경로와 .jpg 파일이 있는지 확인해주세요!")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 실전용 세팅: 배치사이즈 16, num_workers 0 (윈도우 환경 에러 방지)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # 학습 실행
    steps = 4
    epochs = 40
    trainer = Trainer(steps=steps, device=device, train_loader=train_loader, val_loader=val_loader)

    print("Start Training...")
    history = trainer.run(epochs=epochs)
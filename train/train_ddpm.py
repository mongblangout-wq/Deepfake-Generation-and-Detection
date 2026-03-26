"""train/train_ddpm.py"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms

# 앞서 분리한 모듈들을 임포트합니다.
from models.ddpm import DDPMModel
from data.dataset import CelebADataset

# =========================================================================
# 1. Diffusion 파라미터 및 노이즈 추가 함수 (Forward Process)
# =========================================================================
def get_beta_alpha_linear(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    # DDPM 학습 및 샘플링에 쓰일 alpha, beta, alphas_cumprod 반환
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
    betas = torch.tensor(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod

def q_sample(x0, t, noise, alphas_cumprod):
    # 원본 이미지 x0를 입력으로 받아 t스텝만큼 diffusion process 진행
    alpha_bar = alphas_cumprod[t-1].to(x0.device)
    B = alpha_bar.size(0)
    alpha_bar = alpha_bar.reshape((B, 1, 1, 1))

    # x_t가 x0를 평균으로 하는 정규분포를 따름을 이용
    x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t

# =========================================================================
# 2. 학습 및 검증(Train & Test) 루프 정의
# =========================================================================
def train_epoch(model, train_loader, alphas_cumprod, device, optimizer, use_gradient_clipping=True):
    train_loss_sum = 0.0
    train_loss_cnt = 0

    model.train()
    for image, _ in tqdm(train_loader, desc='Training'):
        image = image.to(device)
        eps = torch.randn(image.shape).to(device)

        # 1~1000 범위의 랜덤 t 생성
        t = torch.randint(1, 1001, (image.size(0),), dtype=torch.long).to(device)

        # t스텝만큼 노이즈 추가된 이미지 x_t 생성
        x_t = q_sample(image, t, eps, alphas_cumprod)

        # 예측된 노이즈(eps_theta)와 실제 노이즈(eps)의 MSE Loss 계산
        eps_theta = model(x_t, t)
        loss = F.mse_loss(eps_theta, eps)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # gradient_clipping을 통한 학습 안정화
        if use_gradient_clipping:
            max_grad_norm = 1.0
            clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        train_loss_sum += loss.item()
        train_loss_cnt += 1

    return train_loss_sum / train_loss_cnt

def eval_epoch(model, test_loader, alphas_cumprod, device):
    test_loss_sum = 0.0
    test_loss_cnt = 0

    model.eval()
    with torch.no_grad():
        for image, _ in tqdm(test_loader, desc='Evaluating'):
            image = image.to(device)
            eps = torch.randn(image.shape).to(device)
            t = torch.randint(1, 1001, (image.size(0),), dtype=torch.long).to(device)

            x_t = q_sample(image, t, eps, alphas_cumprod)
            eps_theta = model(x_t, t)
            loss = F.mse_loss(eps_theta, eps)

            test_loss_sum += loss.item()
            test_loss_cnt += 1

    return test_loss_sum / test_loss_cnt

def train(model, train_loader, test_loader, alphas_cumprod, device, optimizer, num_epochs=50, save_model_cycle=10, use_gradient_clipping=True, save_dir="./checkpoints"):
    train_losses = []
    test_losses = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(1, num_epochs+1):
        train_loss = train_epoch(model, train_loader, alphas_cumprod, device, optimizer, use_gradient_clipping)
        test_loss = eval_epoch(model, test_loader, alphas_cumprod, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # 설정한 주기마다 모델 체크포인트 저장
        if epoch % save_model_cycle == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
            }
            model_path = os.path.join(save_dir, f"model_{epoch}.pth")
            torch.save(checkpoint, model_path)
            print(f"Checkpoint saved: {model_path}")

    return train_losses, test_losses

# =========================================================================
# 3. 메인 실행 (Main)
# =========================================================================
if __name__ == '__main__':
    # 1. 경로 설정 문제 해결을 위한 sys.path 추가
    import sys

    BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(BASE_DIR)

    # SEED 고정
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 경로 절대 경로로 설정
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'img_align_celeba')

    # 🚀 실전용 세팅: 전체 데이터 30,000장, 배치사이즈 64
    num_data = 30000
    train_test_ratio = 0.9
    batch_size = 64

    train_transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # 리스트 생성 (1~30000)
    train_img_num = np.arange(1, num_data + 1)[:int(num_data * train_test_ratio)]
    test_img_num = np.arange(1, num_data + 1)[int(num_data * train_test_ratio):]

    # DataLoader 준비 (img_num_list를 사용해 정확히 분할)
    train_dataset = CelebADataset(DATA_DIR, img_num_list=train_img_num, transform=train_transform)
    test_dataset = CelebADataset(DATA_DIR, img_num_list=test_img_num, transform=test_transform)

    # train_loader는 drop_last=True를 주어 배치 사이즈가 안 맞는 마지막 배치를 버림 (에러 방지)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    # DDPM 파라미터 세팅
    _, _, alphas_cumprod = get_beta_alpha_linear()
    alphas_cumprod = alphas_cumprod.to(device)

    # 모델 초기화
    from models.ddpm import DDPMModel

    model = DDPMModel(
        ch=64,
        ch_mult=(1, 2, 4),
        num_res_blocks=2,
        attn_resolutions={32},
        dropout=0.0,
        resamp_with_conv=False,
        init_resolution=64
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # 🚀 실전용 세팅: 50 Epoch 실행, 10번마다 저장
    print("Start DDPM Training...")
    train_losses, test_losses = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        alphas_cumprod=alphas_cumprod,
        device=device,
        optimizer=optimizer,
        num_epochs=50,
        save_model_cycle=10,
        use_gradient_clipping=True
    )
import os
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
from models.ddpm import DDPMModel  # 🚨 반드시 최신 모델을 임포트하세요

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_beta_alpha_linear(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def generate_ddpm_images(checkpoint_path, num_samples=16, save_path='./data/fake_images/ddpm'):
    # 현재 train_ddpm.py의 기본 설정값과 동일하게 맞춥니다.
    model = DDPMModel(
        ch=64, ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions={32}, init_resolution=64
    ).to(device)

    # 가중치 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 저장 방식에 따라 키 이름이 다를 수 있어 예외 처리를 넣었습니다.
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    betas, alphas, alphas_cumprod = get_beta_alpha_linear()
    os.makedirs(save_path, exist_ok=True)

    print(f"Generating images using {checkpoint_path}...")
    with torch.no_grad():
        x = torch.randn(num_samples, 3, 64, 64, device=device)
        for t in tqdm(reversed(range(1000)), total=1000, desc="Denoising"):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
            eps_theta = model(x, t_tensor)

            beta_t = betas[t].to(device)
            alpha_t = alphas[t].to(device)
            alpha_bar_t = alphas_cumprod[t].to(device)

            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta)
            if t > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

        # [ -1, 1 ] 범위를 [ 0, 1 ]로 복원
        generated_images = (x + 1) / 2
        vutils.save_image(generated_images.clamp(0, 1), os.path.join(save_path, 'ddpm_model_1_result.png'), nrow=4)
    print(f"Generation complete! Check: {save_path}")


if __name__ == "__main__":
    # 학습한 model_n.pth 경로를 지정하세요.
    ckpt_path = "./train/checkpoints/model_n.pth"
    if os.path.exists(ckpt_path):
        generate_ddpm_images(ckpt_path)
    else:
        print(f"Error: {ckpt_path} 파일을 찾을 수 없습니다.")
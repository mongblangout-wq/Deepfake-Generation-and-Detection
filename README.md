# Deepfake Generation and Detection Pipeline

본 프로젝트는 ProGAN 및 DDPM/DDIM 알고리즘을 활용하여 고품질의 가상 얼굴 이미지를 생성하고, 이를 정밀하게 판별할 수 있는 Deepfake Detection 모델을 구축하는 엔드투엔드(End-to-End) 파이프라인 프로젝트입니다.

---

## 1. Project Overview

* **목적**: 실제 얼굴 데이터와 구분이 어려운 가상 이미지를 생성하고, 이를 다양한 환경에서 정확하게 탐지하는 판별 모델 구현 및 성능 검증
* **생성 모델(Generation)**: ProGAN (Progressive Growing of GANs), DDPM (Denoising Diffusion Probabilistic Models)
* **탐지 모델(Detection)**: Pre-trained Vision Transformer (ViT) 기반 Fine-tuning
* **데이터셋**: CelebA, FFHQ 기반 고해상도 이미지 및 웹 크롤링 데이터 활용
* **평가 지표**:
    * **Generation**: FID (Frechet Inception Distance), LPIPS (Learned Perceptual Image Patch Similarity)
    * **Detection**: Accuracy, Precision, Recall, F1-score

---

## 2. Key Features

### [Data Acquisition & Preprocessing]
* **Dynamic Web Crawling**: Selenium을 활용한 동적 이미지 수집 파이프라인 구축
* **Face Alignment & Filtering**: MTCNN(Multi-task Cascaded Convolutional Networks)을 활용하여 이미지 내 얼굴을 검출하고, 정면 얼굴(Frontal Face) 데이터만을 선별하여 224x224 및 64x64 해상도로 정규화

### [Generative Modeling]
* **ProGAN Implementation**:
    * 4x4부터 단계적으로 해상도를 높이는 Progressive Growing 구조를 통해 학습 안정성 확보
    * WSConv2d(Weight Scaled Convolution), PixelNorm, MinibatchStd 레이어 구현을 통한 Mode Collapse 방지 및 고품질 이미지 생성
* **DDPM Implementation**:
    * Time Embedding 및 Attention/ResNet Block 기반의 U-Net 구조 설계
    * 1,000 Step의 Diffusion Process를 통해 정밀한 노이즈 복원 모델 구현

### [Deepfake Detection]
* **ViT-based Classification**: 사전 학습된 Vision Transformer 모델을 활용한 Feature Extraction 및 이진 분류(Real/Fake) 수행
* **Optimization**: 생성된 데이터와 실제 데이터를 결합하여 학습을 진행하고, 최적의 Threshold 탐색을 통해 탐지율(Recall) 극대화

---

## 3. Tech Stack

| Category | Details |
| :--- | :--- |
| **Framework** | PyTorch |
| **Generative AI** | ProGAN, DDPM, DDIM |
| **Computer Vision** | OpenCV, PIL, MTCNN, Vision Transformer (ViT) |
| **Analysis & Eval** | Scikit-learn, pytorch-fid, lpips |
| **Automation** | Selenium, WebDriver Manager |

---

## 4. Repository Structure

```text
.
├── data/
│   ├── crawling.py         # Selenium 기반 이미지 수집 및 MTCNN 전처리
│   └── dataset.py          # CelebA 및 크롤링 데이터 커스텀 데이터셋 정의
├── models/
│   ├── progan.py           # ProGAN Generator & Discriminator 구조
│   ├── ddpm.py             # U-Net 기반 Diffusion 모델 구조
│   └── detector.py         # ViT 기반 Deepfake Detector 클래스
├── train/
│   ├── train_progan.py     # ProGAN 단계별 학습 스크립트
│   └── train_ddpm.py       # DDPM 학습 및 체크포인트 저장 스크립트
├── eval/
│   ├── evaluate.py         # FID, LPIPS 이미지 품질 평가 지표 산출
│   └── inference.py        # Detection 모델 추론 및 혼동 행렬 지표 산출
└── .gitignore              # 대용량 데이터셋 및 가중치 파일 제외 설정
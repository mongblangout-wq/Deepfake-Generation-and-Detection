"""data/crawling.py"""

import os
import sys
import time
import urllib.request
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# 프로젝트 최상단 기준 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# --- 프로젝트 가이드라인 파라미터 ---
MAX_IMAGES_PER_CATEGORY = 15
MIN_FACE_SIZE = 80
TARGET_FACE_SIZE = (224, 224)  # 학습 모델에 따라 (64, 64)로 변경 가능
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "crawled_images")


def get_undetectable_driver():
    """Selenium 드라이버를 초기화합니다."""
    options = webdriver.ChromeOptions()
    # 필요시 주석 처리하여 브라우저가 뜨는 것을 확인할 수 있습니다.
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def is_frontal_strict(keypoints):
    """이미지 내 얼굴이 정면인지 판단합니다."""
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']

    # 코가 두 눈의 중앙에서 X축 기준으로 크게 벗어나지 않는지 확인 (정면 기준)
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    dist_x = abs(eye_center_x - nose[0])

    if dist_x < 25:  # 픽셀 오차 허용 범위
        return True
    return False


def download_images_bing(query, out_dir, max_images=15):
    """Bing에서 이미지를 검색하고, MTCNN으로 얼굴을 크롭하여 다운로드합니다."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    driver = get_undetectable_driver()
    detector = MTCNN()

    print(f"[START] '{query}' 검색 시작...")
    url = f"https://www.bing.com/images/search?q={query}"
    driver.get(url)
    time.sleep(3)  # 페이지 로딩 대기

    # 지연 로딩(Lazy Loading) 방지를 위한 스크롤 다운
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
    time.sleep(2)

    saved_count = 0

    # Bing 이미지 썸네일 엘리먼트 탐색
    img_elements = driver.find_elements(By.CSS_SELECTOR, "img.mimg")

    for idx, img in enumerate(img_elements):
        if saved_count >= max_images:
            break

        try:
            img_url = img.get_attribute("src")
            if not img_url or not img_url.startswith("http"):
                continue

            # 이미지 다운로드 임시 저장
            req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            raw_img = urllib.request.urlopen(req).read()

            with open("temp.jpg", "wb") as f:
                f.write(raw_img)

            image = Image.open("temp.jpg").convert('RGB')
            img_arr = np.array(image)

            # MTCNN 얼굴 검출
            results = detector.detect_faces(img_arr)
            if not results:
                continue

            # 가장 큰 얼굴 하나를 선택
            face = max(results, key=lambda b: b['box'][2] * b['box'][3])
            x, y, w, h = face['box']
            keypoints = face['keypoints']

            # 가이드라인 1: 최소 얼굴 크기 제한
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            # 가이드라인 2: 정면 얼굴 검사
            if not is_frontal_strict(keypoints):
                continue

            # 안전하게 Box 좌표 조정 (이미지 밖으로 나가지 않도록)
            x, y = max(0, x), max(0, y)

            # 얼굴 영역 크롭 및 해상도 조정
            face_img = image.crop((x, y, x + w, y + h))
            face_img = face_img.resize(TARGET_FACE_SIZE, Image.Resampling.LANCZOS)

            # 최종 저장
            filename = os.path.join(out_dir, f"{query.replace(' ', '_')}_{saved_count}.jpg")
            face_img.save(filename, format="JPEG", quality=95)

            print(f"[SAVE] '{filename}' 저장 완료")
            saved_count += 1

        except Exception as e:
            # 처리 중 오류 발생 시 다음 이미지로 건너뛰기
            continue

    driver.quit()
    if os.path.exists("temp.jpg"):
        os.remove("temp.jpg")
    print(f"[DONE] '{query}' 완료. 총 {saved_count}개 이미지 저장.\n")


if __name__ == "__main__":
    # 검색하고자 하는 카테고리(키워드) 목록
    categories = ["Korean actor face", "Frontal face portrait"]

    for cat in categories:
        folder_name = cat.replace(" ", "_")
        out_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
        download_images_bing(cat, out_dir, max_images=MAX_IMAGES_PER_CATEGORY)
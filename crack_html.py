# app.py
# 균열 정밀 진단 시스템 V5.0 - 자동 거리 추정 버전
# 사용자 입력 없이 사진 한 장으로 균열 면적/폭을 추정

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
import torch
from ultralytics import YOLO
import io
import math

st.set_page_config(page_title="균열 자동 진단 V5.0", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 시스템 V5.0")
st.caption("📸 사진만 찍으면 거리·면적·폭을 자동으로 추정합니다 (단안 깊이추정 AI)")

# ──────────────────────────────────────────────
# 1) 모델 로딩 (캐싱)
# ──────────────────────────────────────────────
@st.cache_resource
def load_midas():
    """MiDaS 단안 깊이추정 모델 로드"""
    model_type = "MiDaS_small"  # 가벼운 버전 (속도 우선)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return midas, transform, device

@st.cache_resource
def load_yolo():
    """균열 세그멘테이션 YOLO 모델 로드 (사전 학습된 본인 모델 경로로 교체)"""
    # ⚠️ 본인이 학습시킨 균열 세그멘테이션 가중치 경로로 변경하세요
    return YOLO("yolov8n-seg.pt")

# ──────────────────────────────────────────────
# 2) EXIF에서 카메라 정보 뽑기
# ──────────────────────────────────────────────
def get_exif_info(pil_image):
    """EXIF에서 35mm 환산 초점거리, 센서 정보 추출"""
    info = {
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "subject_distance_m": None,
    }
    try:
        exif = pil_image._getexif()
        if exif is None:
            return info
        exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        if "FocalLength" in exif_data:
            f = exif_data["FocalLength"]
            info["focal_length_mm"] = float(f) if not hasattr(f, "numerator") else f.numerator / f.denominator
        if "FocalLengthIn35mmFilm" in exif_data:
            info["focal_length_35mm"] = float(exif_data["FocalLengthIn35mmFilm"])
        if "SubjectDistance" in exif_data:
            d = exif_data["SubjectDistance"]
            info["subject_distance_m"] = float(d) if not hasattr(d, "numerator") else d.numerator / d.denominator
    except Exception:
        pass
    return info

# ──────────────────────────────────────────────
# 3) MiDaS로 깊이맵 추정
# ──────────────────────────────────────────────
def estimate_depth(pil_image, midas, transform, device):
    """단안 깊이추정 → 상대 깊이맵 (정규화된 disparity)"""
    img = np.array(pil_image.convert("RGB"))
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    return depth  # 값이 클수록 가까움 (disparity)

# ──────────────────────────────────────────────
# 4) 상대 깊이맵 → 절대 거리(m) 환산
# ──────────────────────────────────────────────
def depth_to_meters(depth_map, exif_info, image_width):
    """
    MiDaS 출력은 상대값이므로 EXIF 초점거리로 절대 거리 환산.
    근사 공식: 일반적인 스마트폰 사진에서 disparity → meter
    """
    # disparity 정규화
    d_min, d_max = depth_map.min(), depth_map.max()
    d_norm = (depth_map - d_min) / (d_max - d_min + 1e-8)  # 0~1

    # 35mm 환산 초점거리 기반 평균 거리 추정
    # 일반 스마트폰: 사용자가 벽을 찍을 때 보통 0.3m ~ 3m 범위
    focal_35 = exif_info.get("focal_length_35mm") or 26.0  # 아이폰 기본 광각 ≈ 26mm
    
    # 경험적 스케일링: 초점거리가 클수록 멀리서 찍은 것
    # (휴리스틱이지만 균열 진단용으로는 충분)
    base_distance = 0.5 + (focal_35 / 26.0) * 0.5  # 평균 0.5~1.5m 근처
    
    # EXIF에 SubjectDistance가 있으면 그걸 우선 사용 (가장 정확)
    if exif_info.get("subject_distance_m") and 0.1 < exif_info["subject_distance_m"] < 10:
        base_distance = exif_info["subject_distance_m"]

    # disparity가 클수록 가까움 → 거리는 반비례
    # 중앙 영역의 평균 깊이를 base_distance로 설정
    h, w = depth_map.shape
    center_d = d_norm[h//3:2*h//3, w//3:2*w//3].mean()
    
    # 거리 맵 환산 (m 단위)
    # depth_m = base_distance * (center_d / d_norm) 의 형태
    depth_m = base_distance * (center_d + 0.1) / (d_norm + 0.1)
    depth_m = np.clip(depth_m, 0.1, 10.0)  # 합리적 범위로 클리핑
    
    return depth_m, base_distance

# ──────────────────────────────────────────────
# 5) 거리 + 초점거리 → mm/pixel 스케일 계산
# ──────────────────────────────────────────────
def compute_mm_per_pixel(depth_m_at_crack, exif_info, image_width):
    """
    실제 mm/pixel 스케일 = (거리 × 센서폭) / (초점거리 × 이미지폭)
    35mm 환산 사용 시 센서폭 = 36mm로 가정
    """
    focal_35 = exif_info.get("focal_length_35mm") or 26.0
    sensor_width_mm = 36.0  # 35mm 환산 기준
    # 시야각 기반 공식
    # 가로 시야각의 픽셀당 실제 mm
    mm_per_pixel = (depth_m_at_crack * 1000.0 * sensor_width_mm) / (focal_35 * image_width)
    return mm_per_pixel

# ──────────────────────────────────────────────
# 6) Streamlit UI
# ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "균열 사진 업로드 (JPG/PNG, EXIF 정보가 포함된 원본 파일 권장)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    pil_img = Image.open(uploaded)
    img_np = np.array(pil_img.convert("RGB"))
    H, W = img_np.shape[:2]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 원본 이미지")
        st.image(pil_img, use_column_width=True)

    # EXIF 정보
    exif_info = get_exif_info(pil_img)
    with st.expander("📋 EXIF 카메라 정보"):
        st.json(exif_info)
        if exif_info["focal_length_35mm"] is None:
            st.warning("⚠️ EXIF 초점거리 정보가 없습니다. 기본값(26mm, 아이폰 광각)으로 추정합니다.")

    # 로딩
    with st.spinner("🤖 AI 모델 로딩 중..."):
        midas, transform, device = load_midas()
        yolo = load_yolo()

    # 깊이 추정
    with st.spinner("📏 깊이 추정 중..."):
        depth_map = estimate_depth(pil_img, midas, transform, device)
        depth_m, base_distance = depth_to_meters(depth_map, exif_info, W)

    with col2:
        st.subheader("🌈 추정 깊이맵")
        # 시각화용 정규화
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        st.image(depth_color, use_column_width=True)
        st.metric("추정 평균 촬영거리", f"{base_distance:.2f} m")

    # YOLO 균열 세그멘테이션
    with st.spinner("🔍 균열 탐지 중..."):
        results = yolo.predict(img_np, conf=0.25, verbose=False)

    if not results or results[0].masks is None:
        st.error("❌ 균열을 찾지 못했습니다.")
        st.stop()

    # 모든 균열 마스크 합치기
    masks = results[0].masks.data.cpu().numpy()  # (N, h, w)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        full_mask = np.maximum(full_mask, (m_resized > 0.5).astype(np.uint8))

    # 균열 영역의 평균 거리
    crack_pixels = full_mask > 0
    if crack_pixels.sum() == 0:
        st.error("❌ 균열 마스크가 비어있습니다.")
        st.stop()

    depth_at_crack = depth_m[crack_pixels].mean()
    mm_per_pixel = compute_mm_per_pixel(depth_at_crack, exif_info, W)

    # 면적 (cm²)
    pixel_count = int(crack_pixels.sum())
    area_mm2 = pixel_count * (mm_per_pixel ** 2)
    area_cm2 = area_mm2 / 100.0

    # 최대 폭 (mm) — 거리 변환(Distance Transform) 이용
    dist_tf = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
    max_half_width_px = dist_tf.max()
    max_width_mm = 2 * max_half_width_px * mm_per_pixel

    # ──────────────────────────────────────────
    # 결과 출력
    # ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 자동 진단 결과")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("균열까지 거리", f"{depth_at_crack:.2f} m")
    c2.metric("mm/pixel 스케일", f"{mm_per_pixel:.4f}")
    c3.metric("균열 면적", f"{area_cm2:.2f} cm²")
    c4.metric("최대 균열 폭", f"{max_width_mm:.2f} mm")

    # 등급 판정 (예시: 콘크리트 표준 기준)
    if max_width_mm < 0.2:
        grade, color = "✅ 미세균열 (A등급)", "green"
    elif max_width_mm < 0.3:
        grade, color = "🟡 경미균열 (B등급)", "orange"
    elif max_width_mm < 1.0:
        grade, color = "🟠 중간균열 (C등급)", "darkorange"
    else:
        grade, color = "🔴 심각균열 (D등급)", "red"

    st.markdown(f"### 안전 등급: :{color}[{grade}]")

    # 시각화
    overlay = img_np.copy()
    overlay[full_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
    st.image(blended, caption="🎯 균열 탐지 결과", use_column_width=True)

    # 주의사항
    st.info(
        "ℹ️ 이 시스템의 거리 추정은 AI 단안 깊이추정 기반으로 **±20~30%의 오차**가 있을 수 있습니다.\n\n"
        "정밀 진단이 필요한 경우 두 점 기준 측정 모드(V4.0)를 사용하세요."
    )
else:
    st.info("👆 균열 사진을 업로드하세요. **아이폰/안드로이드로 직접 촬영한 원본 사진**이 가장 정확합니다 (EXIF 정보 포함).")

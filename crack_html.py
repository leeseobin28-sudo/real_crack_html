# app.py
# 콘크리트 균열 자동 진단 V5.2 - 배포용 (Streamlit Cloud + 모바일)
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
from ultralytics import YOLO

# HEIC(아이폰) 지원
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

st.set_page_config(page_title="균열 자동 진단", layout="wide")
st.title("🔍 콘크리트 균열 자동 진단 V5.2")
st.caption("📸 사진만 찍거나 업로드하면 거리·면적·폭을 자동 계산합니다.")

# ════════════════════════════════════════════════════════════════
# 1. YOLO 모델 로딩
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_yolo():
    return YOLO("bestcrack.pt")  # ✅ 본인 균열 가중치

# ════════════════════════════════════════════════════════════════
# 2. EXIF 정보 추출
# ════════════════════════════════════════════════════════════════
def get_exif(pil_img):
    info = {"focal_35mm": None, "subject_distance_m": None, "make": None, "model": None}
    try:
        exif = pil_img._getexif()
        if exif is None:
            return info
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        if "FocalLengthIn35mmFilm" in tags:
            info["focal_35mm"] = float(tags["FocalLengthIn35mmFilm"])
        if "SubjectDistance" in tags:
            d = tags["SubjectDistance"]
            info["subject_distance_m"] = float(d) if not hasattr(d, "numerator") else d.numerator / d.denominator
        info["make"] = tags.get("Make", None)
        info["model"] = tags.get("Model", None)
    except Exception:
        pass
    return info

# ════════════════════════════════════════════════════════════════
# 3. 거리 자동 추정
# ════════════════════════════════════════════════════════════════
def estimate_distance(exif_info):
    if exif_info["subject_distance_m"] and 0.1 < exif_info["subject_distance_m"] < 10:
        return exif_info["subject_distance_m"], "EXIF SubjectDistance"
    return 0.8, "기본 가정 (0.8m)"

# ════════════════════════════════════════════════════════════════
# 4. mm/pixel 계산
# ════════════════════════════════════════════════════════════════
def mm_per_pixel(dist_m, focal_35mm, image_width_px):
    f = focal_35mm or 26.0
    return (dist_m * 1000.0 * 36.0) / (f * image_width_px)

# ════════════════════════════════════════════════════════════════
# 5. 입력 UI (카메라 + 업로더)
# ════════════════════════════════════════════════════════════════
mode = st.radio(
    "📥 입력 방식 선택",
    ["📷 카메라로 촬영", "📁 파일 업로드"],
    horizontal=True
)

uploaded = None
if mode == "📷 카메라로 촬영":
    uploaded = st.camera_input("균열을 촬영하세요")
else:
    uploaded = st.file_uploader(
        "균열 사진 업로드",
        type=["jpg", "jpeg", "png", "heic", "heif"]
    )

# 사이드바: 수동 보정 옵션
st.sidebar.header("⚙️ 보정 옵션")
manual_dist = st.sidebar.slider(
    "촬영거리 수동 조정 (m) — 0이면 자동",
    min_value=0.0, max_value=3.0, value=0.0, step=0.1,
)
conf_thres = st.sidebar.slider("YOLO 신뢰도 임계값", 0.05, 0.9, 0.25, 0.05)

if not uploaded:
    st.info("👆 사진을 촬영하거나 업로드하세요.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# 6. 이미지 로딩
# ════════════════════════════════════════════════════════════════
try:
    pil_img = Image.open(uploaded)
    pil_img = pil_img.convert("RGB")
except Exception as e:
    st.error(f"❌ 이미지 로드 실패: {e}")
    st.stop()

img_np = np.array(pil_img)
H, W = img_np.shape[:2]

col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 입력 이미지")
    st.image(pil_img, use_container_width=True)

# EXIF
exif_info = get_exif(pil_img)
with col2:
    st.subheader("📋 카메라 정보")
    st.write(f"📱 기기: **{exif_info['make']} {exif_info['model']}**")
    st.write(f"🔭 초점거리(35mm): **{exif_info['focal_35mm']} mm**")
    st.write(f"📏 EXIF 거리: **{exif_info['subject_distance_m']} m**")
    if mode == "📷 카메라로 촬영":
        st.warning("⚠️ 카메라 촬영 모드는 EXIF가 제한적입니다. 결과가 어색하면 사이드바에서 거리를 조정하세요.")

# 거리 추정
auto_dist, source = estimate_distance(exif_info)
if manual_dist > 0:
    dist = manual_dist
    source = "수동 입력"
else:
    dist = auto_dist

# ════════════════════════════════════════════════════════════════
# 7. YOLO 추론
# ════════════════════════════════════════════════════════════════
with st.spinner("🔍 균열 탐지 중..."):
    yolo = load_yolo()
    results = yolo.predict(img_np, conf=conf_thres, verbose=False)

if not results or results[0].masks is None:
    st.error("❌ 균열을 찾지 못했습니다. 사이드바에서 신뢰도 임계값을 낮춰보세요.")
    st.stop()

masks = results[0].masks.data.cpu().numpy()
full_mask = np.zeros((H, W), dtype=np.uint8)
for m in masks:
    mr = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    full_mask = np.maximum(full_mask, (mr > 0.5).astype(np.uint8))

if full_mask.sum() == 0:
    st.error("❌ 균열 마스크가 비어있습니다.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# 8. 면적·폭 계산
# ════════════════════════════════════════════════════════════════
scale = mm_per_pixel(dist, exif_info["focal_35mm"], W)
pixel_cnt = int(full_mask.sum())
area_cm2 = (pixel_cnt * scale * scale) / 100.0
dt = cv2.distanceTransform(full_mask, cv2.DIST_L2, 5)
max_width_mm = 2 * float(dt.max()) * scale

# ════════════════════════════════════════════════════════════════
# 9. 결과 출력
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 자동 진단 결과")

c1, c2, c3, c4 = st.columns(4)
c1.metric("촬영거리", f"{dist:.2f} m", help=f"출처: {source}")
c2.metric("mm/pixel", f"{scale:.4f}")
c3.metric("균열 면적", f"{area_cm2:.2f} cm²")
c4.metric("최대 균열 폭", f"{max_width_mm:.2f} mm")

if max_width_mm < 0.2:
    grade, color = "✅ 미세균열 (A등급)", "green"
elif max_width_mm < 0.3:
    grade, color = "🟡 경미균열 (B등급)", "orange"
elif max_width_mm < 1.0:
    grade, color = "🟠 중간균열 (C등급)", "darkorange"
else:
    grade, color = "🔴 심각균열 (D등급)", "red"
st.markdown(f"### 안전 등급: :{color}[{grade}]")

overlay = img_np.copy()
overlay[full_mask > 0] = [255, 50, 50]
blended = cv2.addWeighted(img_np, 0.55, overlay, 0.45, 0)
st.image(blended, caption="🎯 균열 탐지 결과", use_container_width=True)

st.info(
    "ℹ️ 거리 자동 추정은 휴리스틱 기반입니다.\n\n"
    "🔧 결과가 어색하면 사이드바에서 거리·신뢰도를 조정하세요."
)

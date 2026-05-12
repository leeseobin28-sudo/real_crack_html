import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import math
from streamlit_drawable_canvas import st_canvas

# ============================================================
# 1. 초기 설정
# ============================================================
st.set_page_config(page_title="콘크리트 균열 정밀 진단 V4", page_icon="🏗️", layout="wide")
st.title("🏗️ 콘크리트 균열 정밀 진단 V4.0")
st.caption("아이폰 측정앱 방식: 사진 위에서 **시작점 → 끝점**을 찍으면 그 구간의 균열을 mm 단위로 분석합니다.")

# HEIC 지원
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

# ============================================================
# 2. 모델 로드
# ============================================================
@st.cache_resource
def load_model():
    for p in ["bestcrack.pt", os.path.expanduser("~/Desktop/bestcrack.pt")]:
        if os.path.exists(p):
            return YOLO(p)
    return None

model_crack = load_model()
if model_crack is None:
    st.error("❌ bestcrack.pt 모델 파일을 찾을 수 없습니다.")
    st.stop()

# ============================================================
# 3. 세션 상태
# ============================================================
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ============================================================
# 4. 분석 함수 (두 점 사이 ROI 영역의 균열만 분석)
# ============================================================
def analyze_crack_between_points(image_np, p1, p2, real_length_cm,
                                  conf_threshold=0.25, dilation_iter=1):
    """
    p1, p2          : 원본 이미지 픽셀 좌표 (x, y)
    real_length_cm  : 사용자가 입력한 p1-p2의 실제 거리 (cm)
    """
    draw_img = image_np.copy()
    overlay = image_np.copy()

    # ── 1) 픽셀 거리 → mm 스케일 (FOV 추정 불필요, 정확함) ──
    px_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if px_dist < 5:
        return draw_img, 0, 0, 0, "두 점이 너무 가깝습니다."

    mm_per_pixel = (real_length_cm * 10.0) / px_dist
    cm_per_pixel = mm_per_pixel / 10.0

    # ── 2) YOLO 균열 탐지 ──
    results = model_crack.predict(source=image_np, conf=conf_threshold, verbose=False)
    if not results or results[0].masks is None:
        cv2.line(draw_img, p1, p2, (0, 255, 255), 3)
        cv2.circle(draw_img, p1, 10, (0, 255, 0), -1)
        cv2.circle(draw_img, p2, 10, (0, 0, 255), -1)
        return draw_img, 0, 0, mm_per_pixel, "균열이 탐지되지 않았습니다."

    result = results[0]

    # ── 3) 전체 균열 마스크 ──
    H, W = image_np.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.uint8)
    for m in result.masks.xy:
        if len(m) > 0:
            pts = np.array(m, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(full_mask, [pts], 1)

    # ── 4) 팽창 교정 ──
    if dilation_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        full_mask = cv2.dilate(full_mask, kernel, iterations=dilation_iter)

    # ── 5) 두 점을 잇는 띠(ROI) 생성 ──
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    band_thickness = max(int(px_dist * 0.3), 40)  # 균열이 곧지 않을 수 있어 여유
    cv2.line(roi_mask, p1, p2, 1, thickness=band_thickness * 2)

    # ── 6) ROI ∩ 균열 ──
    target_mask = full_mask & roi_mask

    # ── 7) 면적 / 최대폭 ──
    total_px = int(np.sum(target_mask))
    area_cm2 = total_px * (cm_per_pixel ** 2)

    max_thick_mm = 0
    if total_px > 0:
        dist_tf = cv2.distanceTransform(target_mask, cv2.DIST_L2, 3)
        max_thick_mm = float(np.max(dist_tf)) * 2 * mm_per_pixel

    # ── 8) 시각화 ──
    overlay[roi_mask == 1] = (255, 255, 0)       # ROI: 연노랑
    overlay[target_mask == 1] = (255, 0, 0)      # 균열: 빨강
    cv2.addWeighted(overlay, 0.4, draw_img, 0.6, 0, draw_img)

    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(draw_img, contours, -1, (255, 0, 0), 2)

    # 측정선 + 양 끝점
    cv2.line(draw_img, p1, p2, (0, 255, 255), 3)
    cv2.circle(draw_img, p1, 12, (0, 255, 0), -1)
    cv2.circle(draw_img, p1, 12, (255, 255, 255), 2)
    cv2.circle(draw_img, p2, 12, (0, 0, 255), -1)
    cv2.circle(draw_img, p2, 12, (255, 255, 255), 2)

    # 거리 라벨
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    label = f"{real_length_cm:.1f} cm ({px_dist:.0f}px)"
    cv2.putText(draw_img, label, (mid[0] + 10, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(draw_img, label, (mid[0] + 10, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return draw_img, area_cm2, max_thick_mm, mm_per_pixel, "OK"

# ============================================================
# 5. 이미지 입력
# ============================================================
st.markdown("### 📷 1단계 · 이미지 입력")
types = ["jpg", "jpeg", "png"] + (["heic", "heif"] if HEIC_OK else [])

tab1, tab2 = st.tabs(["📁 업로드", "📸 카메라 촬영"])
with tab1:
    up = st.file_uploader("이미지 선택", type=types)
    if up:
        st.session_state.captured_image = ImageOps.exif_transpose(Image.open(up)).convert("RGB")
        st.session_state.analysis_result = None
with tab2:
    cam = st.camera_input("균열을 정면(90°)에서 촬영")
    if cam:
        st.session_state.captured_image = ImageOps.exif_transpose(Image.open(cam)).convert("RGB")
        st.session_state.analysis_result = None

# ============================================================
# 6. 2점 찍기 + 분석
# ============================================================
if st.session_state.captured_image is not None:
    image = st.session_state.captured_image
    img_np = np.array(image)
    H, W = img_np.shape[:2]

    st.markdown("### 🎯 2단계 · 시작점 → 끝점 찍기")
    st.info(
        "👉 아래 이미지에서 **균열의 시작점을 클릭, 이어서 끝점을 클릭**하세요.\n\n"
        "📏 그 다음 두 점 사이의 **실제 거리(cm)**를 입력합니다. "
        "(자/줄자로 재거나, 옆에 둔 동전·벽돌처럼 길이를 아는 기준물 활용)"
    )

    # 캔버스 표시 크기
    DISPLAY_W = 700
    scale = DISPLAY_W / W
    DISPLAY_H = int(H * scale)

    col_canvas, col_info = st.columns([2, 1])

    with col_canvas:
        canvas_res = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=8,
            stroke_color="#00FF00",
            background_image=image.resize((DISPLAY_W, DISPLAY_H)),
            update_streamlit=True,
            height=DISPLAY_H,
            width=DISPLAY_W,
            drawing_mode="point",
            point_display_radius=8,
            key="canvas",
        )

    with col_info:
        st.markdown("#### ⚙️ 측정 설정")
        real_cm = st.number_input(
            "📏 두 점 사이의 실제 거리 (cm)",
            min_value=0.5, value=10.0, step=0.5,
            help="자로 잰 실제 거리. 정확할수록 결과도 정확합니다.\n"
                 "💡 기준물 예시 — 500원: 2.65cm / 표준벽돌 길이: 19cm / 신용카드 가로: 8.56cm"
        )
        conf_th = st.slider("탐지 민감도", 0.05, 0.95, 0.25, 0.05)
        dilation = st.slider("마스킹 팽창 교정", 0, 5, 1)

        # 찍힌 점 추출
        points = []
        if canvas_res.json_data is not None:
            for o in canvas_res.json_data.get("objects", []):
                if o.get("type") == "circle":
                    cx = (o["left"] + o.get("radius", 0)) / scale
                    cy = (o["top"] + o.get("radius", 0)) / scale
                    points.append((int(cx), int(cy)))

        st.markdown(f"**찍힌 점: {len(points)}개**")
        if len(points) >= 1:
            st.write(f"🟢 시작점: {points[0]}")
        if len(points) >= 2:
            st.write(f"🔴 끝점: {points[1]}")
            px_d = math.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1])
            st.write(f"📐 픽셀 거리: {px_d:.1f} px")
            st.write(f"🔬 스케일: 1px ≈ {(real_cm*10/px_d):.3f} mm")
        if len(points) > 2:
            st.warning("⚠️ 점이 3개 이상입니다. 캔버스 휴지통(🗑️)으로 지우거나, 처음 2개만 사용됩니다.")

        analyze_btn = st.button(
            "🚀 측정 시작",
            type="primary",
            use_container_width=True,
            disabled=(len(points) < 2),
        )

    # ── 분석 실행 ──
    if analyze_btn and len(points) >= 2:
        p1, p2 = points[0], points[1]
        with st.spinner("AI 분석 중..."):
            result_img, area, thick, mmpp, msg = analyze_crack_between_points(
                img_np, p1, p2, real_cm,
                conf_threshold=conf_th,
                dilation_iter=dilation,
            )
            st.session_state.analysis_result = {
                "img": result_img, "area": area, "thick": thick,
                "mmpp": mmpp, "msg": msg, "real_cm": real_cm
            }

    # ── 결과 표시 ──
    if st.session_state.analysis_result is not None:
        st.markdown("---")
        st.markdown("### ✨ 3단계 · 분석 결과")
        r = st.session_state.analysis_result

        st.image(r["img"], use_container_width=True,
                 caption=f"분석 결과 (스케일: 1px ≈ {r['mmpp']:.3f} mm)")

        if r["area"] > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📐 균열 면적", f"{r['area']:.2f} cm²")
            m2.metric("🔥 최대 균열 폭", f"{r['thick']:.2f} mm")
            m3.metric("📏 측정 기준", f"{r['real_cm']:.1f} cm")
            status_lbl = "⚠️ 보수 필요" if r["thick"] >= 0.3 else "✅ 양호"
            m4.metric("📋 안전 진단", status_lbl)

            with st.expander("ℹ️ 균열 폭 진단 기준"):
                st.markdown("""
                - **0.3 mm 미만** : 일반적으로 구조 안전성에 큰 영향 없음 (양호)
                - **0.3 mm 이상** : 철근 부식 우려, 보수 권장
                - **1.0 mm 이상** : 구조적 문제 가능성, 정밀 점검 필요
                """)
        else:
            st.warning(f"⚠️ {r['msg']}")
else:
    st.info("👆 위에서 이미지를 업로드하거나 촬영해주세요.")

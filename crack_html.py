import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import math
import streamlit.components.v1 as components

# 1. 아이폰 HEIC 지원
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

st.set_page_config(page_title="초정밀 균열 진단 AI", page_icon="🏗️", layout="wide")
st.title("🏗️ 콘크리트 균열 정밀 진단 시스템 V4")

# 2. 모델 로드
@st.cache_resource
def load_model():
    # 파일 경로 유연하게 대응
    MODEL_PATH = "bestcrack.pt"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.expanduser("~/Desktop/bestcrack.pt")
    return YOLO(MODEL_PATH)

try:
    model_crack = load_model()
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 3. HTML/JS AR 측정 컴포넌트 (시각화 기능 추가)
def ar_distance_measurer():
    ar_html = """
    <div id="container" style="position: relative; width: 100%; height: 450px; background: #000; border-radius: 20px; overflow: hidden; border: 2px solid #4A4A4A;">
        <video id="video" style="width: 100%; height: 100%; object-fit: cover;" autoplay playsinline></video>
        
        <div id="reticle" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); pointer-events: none; z-index: 10;">
            <div style="width: 30px; height: 30px; border: 2px solid white; border-radius: 50%;"></div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; background: red; border-radius: 50%;"></div>
        </div>

        <div id="startMarker" style="position: absolute; display: none; width: 15px; height: 15px; background: lime; border: 2px solid white; border-radius: 50%; transform: translate(-50%, -50%); z-index: 5;"></div>

        <div style="position: absolute; bottom: 20px; width: 100%; text-align: center; display: flex; justify-content: center; gap: 15px; z-index: 20;">
            <button id="btnAction" style="padding: 12px 24px; background: #FF4B4B; color: white; border: none; border-radius: 30px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">시작점 고정</button>
            <button id="btnReset" style="padding: 12px 24px; background: #555; color: white; border: none; border-radius: 30px; cursor: pointer;">초기화</button>
        </div>
        
        <div id="info" style="position: absolute; top: 15px; left: 15px; color: white; background: rgba(0,0,0,0.7); padding: 8px 15px; border-radius: 10px; font-size: 14px; z-index: 20;">준비됨</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const btnAction = document.getElementById('btnAction');
        const btnReset = document.getElementById('btnReset');
        const info = document.getElementById('info');
        const startMarker = document.getElementById('startMarker');
        
        let isMeasuring = false;
        let startAlpha = 0, startBeta = 0;
        let currentOri = { alpha: 0, beta: 0 };

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(s => { video.srcObject = s; })
            .catch(e => { alert("카메라 접근 권한이 필요합니다."); });

        async function requestSensorPermission() {
            if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
                const state = await DeviceOrientationEvent.requestPermission();
                if (state === 'granted') {
                    window.addEventListener('deviceorientation', e => { currentOri.alpha = e.alpha; currentOri.beta = e.beta; });
                    return true;
                }
                return false;
            }
            window.addEventListener('deviceorientation', e => { currentOri.alpha = e.alpha; currentOri.beta = e.beta; });
            return true;
        }

        btnAction.onclick = async () => {
            await requestSensorPermission();

            if (!isMeasuring) {
                // 시작점 고정
                startAlpha = currentOri.alpha;
                startBeta = currentOri.beta;
                isMeasuring = true;
                
                // 시각적 피드백: 시작점 마커를 현재 중앙에 고정 (가상)
                startMarker.style.display = "block";
                startMarker.style.top = "50%";
                startMarker.style.left = "50%";
                
                btnAction.innerText = "끝점 지정";
                btnAction.style.background = "#007AFF";
                info.innerText = "이동 중...";
            } else {
                // 끝점 계산
                const dAlpha = Math.abs(currentOri.alpha - startAlpha) * (Math.PI/180);
                const dBeta = Math.abs(currentOri.beta - startBeta) * (Math.PI/180);
                const dist = Math.sqrt(Math.pow(dAlpha, 2) + Math.pow(dBeta, 2)) * 1.5; // 보정계수 1.5
                
                info.innerText = "측정 완료: " + dist.toFixed(3) + "m";
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: dist}, '*');
                btnAction.disabled = true;
                btnAction.style.background = "#888";
            }
        };

        btnReset.onclick = () => {
            isMeasuring = false;
            startMarker.style.display = "none";
            btnAction.disabled = false;
            btnAction.innerText = "시작점 고정";
            btnAction.style.background = "#FF4B4B";
            info.innerText = "초기화됨";
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: 0}, '*');
        };
    </script>
    """
    return components.html(ar_html, height=480)

# 4. 분석 함수
def analyze_and_draw(image_np, results_crack, dilation_iter):
    draw_img = image_np.copy()
    overlay = image_np.copy()
    if not results_crack or len(results_crack) == 0:
        return draw_img, 0, 0
    
    result = results_crack[0]
    mask_canvas = np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    if result.masks is not None:
        for mask in result.masks.xy:
            if len(mask) > 0:
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_canvas, [pts], 1)
    
    if dilation_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask_canvas = cv2.dilate(mask_canvas, kernel, iterations=dilation_iter)
    
    overlay[mask_canvas == 1] = (255, 0, 0)
    
    max_thickness_px = 0
    if np.sum(mask_canvas) > 0:
        dist_transform = cv2.distanceTransform(mask_canvas, cv2.DIST_L2, 3)
        max_thickness_px = np.max(dist_transform) * 2 
    
    cv2.addWeighted(overlay, 0.5, draw_img, 0.5, 0, draw_img)
    return draw_img, int(np.sum(mask_canvas)), max_thickness_px

# 5. UI 로직
# TypeError 해결: auto_distance가 0이거나 None일 때 처리 강화
auto_distance_raw = ar_distance_measurer()

# auto_distance_raw 값의 타입을 안전하게 변환
try:
    if auto_distance_raw is None:
        auto_distance = 0.0
    elif isinstance(auto_distance_raw, (int, float)):
        auto_distance = float(auto_distance_raw)
    else:
        # 혹시나 리스트나 다른 객체로 올 경우를 대비
        auto_distance = 0.0
except:
    auto_distance = 0.0

st.markdown("---")
uploaded_file = st.file_uploader("📸 분석할 균열 사진을 업로드하세요", type=["jpg", "png", "jpeg", "heic"])

if uploaded_file is not None:
    # AR에서 측정한 값이 있으면 그 값을 기본값으로, 없으면 1.0m
    default_dist = auto_distance if auto_distance > 0 else 1.0
    
    col_set1, col_set2 = st.columns([2, 1])
    with col_set1:
        final_dist = st.number_input("📏 촬영 거리 (미터 단위)", value=default_dist, step=0.001, format="%.3f")
    with col_set2:
        st.write("") # 간격 조절
        st.write(f"현재 AR 측정값: **{auto_distance:.3f}m**")

    analyze_btn = st.button("🚀 AI 균열 분석 시작", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("AI 엔진 가동 중..."):
            image = ImageOps.exif_transpose(Image.open(uploaded_file))
            img_array = np.array(image.convert("RGB"))
            
            # 스케일 계산 (아이폰 일반적인 FOV 70도 기준)
            fov_radians = math.radians(70 / 2)
            predicted_width_cm = 2 * (final_dist * 100) * math.tan(fov_radians)
            cm_per_px = predicted_width_cm / img_array.shape[1]
            
            res = model_crack.predict(source=img_array, conf=0.25, verbose=False)
            final_img, area_px, thick_px = analyze_and_draw(img_array, res, 1)
            
            st.image(final_img, caption=f"분석 결과 (입력 거리: {final_dist}m)", use_container_width=True)
            
            m1, m2 = st.columns(2)
            if area_px > 0:
                area_cm2 = area_px * (cm_per_px**2)
                thick_mm = thick_px * (cm_per_px*10)
                m1.metric("📐 균열 면적", f"{area_cm2:.2f} cm²")
                m2.metric("🔥 최대 균열 폭", f"{thick_mm:.2f} mm")
            else:
                st.warning("균열이 발견되지 않았습니다. 사진을 더 가까이서 찍거나 밝은 곳에서 촬영해주세요.")
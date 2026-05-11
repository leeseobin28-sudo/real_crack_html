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
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False

st.set_page_config(page_title="초정밀 균열 진단 AI", page_icon="🏗️", layout="wide")
st.title("🏗️ 콘크리트 균열 정밀 진단 시스템 V4")

# 2. 모델 로드
@st.cache_resource
def load_model():
    MODEL_PATH = "bestcrack.pt" if os.path.exists("bestcrack.pt") else os.path.expanduser("~/Desktop/bestcrack.pt")
    return YOLO(MODEL_PATH)

try:
    model_crack = load_model()
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 3. HTML/JS AR 측정 컴포넌트
def ar_distance_measurer():
    ar_html = """
    <div id="container" style="position: relative; width: 100%; height: 450px; background: #000; border-radius: 20px; overflow: hidden; border: 2px solid #4A4A4A;">
        <video id="video" style="width: 100%; height: 100%; object-fit: cover;" autoplay playsinline></video>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); pointer-events: none;">
            <div style="width: 40px; height: 40px; border: 2px solid rgba(255,255,255,0.8); border-radius: 50%;"></div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 4px; height: 4px; background: red; border-radius: 50%;"></div>
        </div>
        <div style="position: absolute; bottom: 20px; width: 100%; text-align: center; display: flex; justify-content: center; gap: 15px;">
            <button id="btnAction" style="padding: 12px 24px; background: #FF4B4B; color: white; border: none; border-radius: 30px; font-weight: bold; cursor: pointer;">시작점 고정</button>
            <button id="btnReset" style="padding: 12px 24px; background: #555; color: white; border: none; border-radius: 30px; cursor: pointer;">초기화</button>
        </div>
        <div id="info" style="position: absolute; top: 15px; left: 15px; color: white; background: rgba(0,0,0,0.6); padding: 8px 15px; border-radius: 10px; font-size: 14px;">거리: 0.00m (준비)</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const btnAction = document.getElementById('btnAction');
        const btnReset = document.getElementById('btnReset');
        const info = document.getElementById('info');
        
        let isMeasuring = false;
        let startAlpha = 0, startBeta = 0;
        let currentOri = { alpha: 0, beta: 0 };

        // 카메라 시작
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(s => { video.srcObject = s; })
            .catch(e => { alert("카메라를 시작할 수 없습니다. 권한을 확인해주세요."); });

        // iOS 센서 권한 요청 함수
        async function requestSensorPermission() {
            if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
                try {
                    const permissionState = await DeviceOrientationEvent.requestPermission();
                    if (permissionState === 'granted') {
                        window.addEventListener('deviceorientation', handleOrientation);
                        return true;
                    } else {
                        alert("센서 권한이 거부되었습니다. 설정에서 허용해주세요.");
                        return false;
                    }
                } catch (e) {
                    alert("권한 요청 중 오류 발생");
                    return false;
                }
            } else {
                window.addEventListener('deviceorientation', handleOrientation);
                return true;
            }
        }

        function handleOrientation(e) {
            currentOri.alpha = e.alpha;
            currentOri.beta = e.beta;
        }

        btnAction.onclick = async () => {
            // 클릭 시 권한 확인 및 요청
            const granted = await requestSensorPermission();
            if (!granted) return;

            if (!isMeasuring) {
                // 시작점 저장
                startAlpha = currentOri.alpha;
                startBeta = currentOri.beta;
                isMeasuring = true;
                btnAction.innerText = "끝점 지정";
                btnAction.style.background = "#007AFF";
                info.innerText = "측정 중... 폰을 끝점으로 천천히 움직이세요.";
            } else {
                // 끝점 계산
                const dAlpha = Math.abs(currentOri.alpha - startAlpha) * (Math.PI/180);
                const dBeta = Math.abs(currentOri.beta - startBeta) * (Math.PI/180);
                const dist = Math.sqrt(Math.pow(dAlpha, 2) + Math.pow(dBeta, 2)) * 1.5;
                
                info.innerText = "계산 완료: " + dist.toFixed(2) + "m";
                // Streamlit으로 데이터 전송
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: dist}, '*');
                btnAction.disabled = true;
                btnAction.style.background = "#888";
            }
        };

        btnReset.onclick = () => {
            isMeasuring = false;
            btnAction.disabled = false;
            btnAction.innerText = "시작점 고정";
            btnAction.style.background = "#FF4B4B";
            info.innerText = "거리: 0.00m (준비)";
            // 리셋 시 0으로 보낼 수도 있음
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: 0}, '*');
        };
    </script>
    """
    return components.html(ar_html, height=500)

# 4. 분석 함수
def analyze_and_draw(image_np, results_crack, dilation_iter):
    draw_img = image_np.copy()
    overlay = image_np.copy()
    if not results_crack or len(results_crack) == 0: return draw_img, 0, 0
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

# 5. UI 및 로직
auto_distance = ar_distance_measurer()

st.markdown("---")
uploaded_file = st.file_uploader("📸 분석할 균열 사진을 업로드하세요", type=["jpg", "png", "jpeg", "heic"])

if uploaded_file is not None:
    # 측정된 값이 있으면 사용, 없으면 1.0m
    initial_val = float(auto_distance) if auto_distance and auto_distance > 0 else 1.0
    final_dist = st.number_input("📏 확정된 측정 거리 (미터)", value=initial_val, step=0.01)
    
    analyze_btn = st.button("🚀 AI 분석 시작", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("AI 분석 중..."):
            image = ImageOps.exif_transpose(Image.open(uploaded_file))
            img_array = np.array(image.convert("RGB"))
            
            # 스케일 계산
            fov_radians = math.radians(70 / 2)
            predicted_width_cm = 2 * (final_dist * 100) * math.tan(fov_radians)
            cm_per_px = predicted_width_cm / img_array.shape[1]
            
            res = model_crack.predict(source=img_array, conf=0.25, verbose=False)
            final_img, area_px, thick_px = analyze_and_draw(img_array, res, 1)
            
            st.image(final_img, use_container_width=True)
            
            col_m1, col_m2 = st.columns(2)
            if area_px > 0:
                area_cm2 = area_px * (cm_per_px**2)
                thick_mm = thick_px * (cm_per_px*10)
                col_m1.metric("📐 균열 면적", f"{area_cm2:.2f} cm²")
                col_m2.metric("🔥 최대 균열 폭", f"{thick_mm:.2f} mm")
            else:
                st.warning("탐지된 균열이 없습니다.")
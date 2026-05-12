import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import math
import streamlit.components.v1 as components
import base64
from io import BytesIO

# 1. 환경 설정
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

st.set_page_config(page_title="AR 실시간 균열 진단", page_icon="🏗️", layout="wide")
st.title("🏗️ AR 실시간 균열 정밀 진단 V5.2")

# 2. 모델 로드
@st.cache_resource
def load_model():
    MODEL_PATH = "bestcrack.pt"
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.expanduser("~/Desktop/bestcrack.pt")
    return YOLO(MODEL_PATH)

try:
    model_crack = load_model()
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

# 3. 분석 함수
def analyze_captured_image(image_np, physical_dist_m, results_crack):
    draw_img = image_np.copy()
    if not results_crack or len(results_crack) == 0:
        return draw_img, 0, 0
    
    result = results_crack[0]
    mask_canvas = np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    if result.masks is not None:
        for mask in result.masks.xy:
            if len(mask) > 0:
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_canvas, [pts], 1)
    
    fov_radians = math.radians(70 / 2)
    real_view_width_cm = 2 * (physical_dist_m * 100) * math.tan(fov_radians)
    cm_per_px = real_view_width_cm / image_np.shape[1]
    
    draw_img[mask_canvas == 1] = [255, 0, 0]
    
    max_thickness_px = 0
    if np.sum(mask_canvas) > 0:
        dist_transform = cv2.distanceTransform(mask_canvas, cv2.DIST_L2, 3)
        max_thickness_px = np.max(dist_transform) * 2
        
    area_cm2 = np.sum(mask_canvas) * (cm_per_px**2)
    thick_mm = max_thickness_px * (cm_per_px * 10)
    
    return draw_img, area_cm2, thick_mm

# 4. AR 카메라 컴포넌트
def ar_scanner_component():
    # 데이터 전송 시 'Streamlit.setComponentValue'가 안정적으로 작동하도록 감싸는 로직
    ar_html = """
    <div id="wrapper" style="position: relative; width: 100%; height: 500px; background: #000; border-radius: 20px; overflow: hidden; border: 3px solid #333;">
        <video id="video" style="width: 100%; height: 100%; object-fit: cover;" autoplay playsinline></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
        
        <div id="ui" style="position: absolute; bottom: 25px; width: 100%; display: flex; justify-content: center; gap: 15px;">
            <button id="mainBtn" style="padding: 15px 25px; background: #FF4B4B; color: white; border: none; border-radius: 30px; font-weight: bold; font-size: 16px; min-width: 140px;">시작점 고정</button>
            <button id="resetBtn" style="padding: 15px 25px; background: #555; color: white; border: none; border-radius: 30px;">초기화</button>
        </div>
        <div id="status" style="position: absolute; top: 15px; left: 15px; color: #00FF00; background: rgba(0,0,0,0.7); padding: 8px 15px; border-radius: 8px; font-size: 13px;">READY</div>
    </div>

    <script>
        // 스트림릿과 통신하기 위한 객체 확인
        function sendMessage(data) {
            if (window.Streamlit) {
                window.Streamlit.setComponentValue(data);
            } else {
                // 스트림릿 객체가 로드될 때까지 재시도
                setTimeout(() => sendMessage(data), 100);
            }
        }

        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const mainBtn = document.getElementById('mainBtn');
        const resetBtn = document.getElementById('resetBtn');
        const status = document.getElementById('status');

        let isMeasuring = false;
        let startOri = null;
        let currentOri = { alpha: 0, beta: 0 };
        let finalDist = 0;

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(s => { video.srcObject = s; });

        async function initSensor() {
            if (window.DeviceOrientationEvent && typeof DeviceOrientationEvent.requestPermission === 'function') {
                const res = await DeviceOrientationEvent.requestPermission();
                if (res === 'granted') {
                    window.addEventListener('deviceorientation', e => { currentOri = {alpha: e.alpha, beta: e.beta}; });
                }
            } else {
                window.addEventListener('deviceorientation', e => { currentOri = {alpha: e.alpha || 0, beta: e.beta || 0}; });
            }
        }

        function draw() {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(centerX, centerY, 15, 0, Math.PI*2); ctx.stroke();

            if (isMeasuring && startOri) {
                const sensitivity = 25; 
                const offsetX = (currentOri.alpha - startOri.alpha) * sensitivity;
                const offsetY = (currentOri.beta - startOri.beta) * sensitivity;
                const startPointX = centerX - offsetX;
                const startPointY = centerY + offsetY;

                ctx.fillStyle = "lime";
                ctx.beginPath(); ctx.arc(startPointX, startPointY, 8, 0, Math.PI*2); ctx.fill();

                ctx.setLineDash([5, 5]);
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.beginPath(); ctx.moveTo(startPointX, startPointY); ctx.lineTo(centerX, centerY); ctx.stroke();
                ctx.setLineDash([]);

                const dAlpha = Math.abs(currentOri.alpha - startOri.alpha) * (Math.PI/180);
                const dBeta = Math.abs(currentOri.beta - startOri.beta) * (Math.PI/180);
                finalDist = Math.sqrt(Math.pow(dAlpha, 2) + Math.pow(dBeta, 2)) * 1.5;
                status.innerText = "거리: " + finalDist.toFixed(3) + "m";
            }
            requestAnimationFrame(draw);
        }
        draw();

        mainBtn.onclick = async () => {
            await initSensor();
            if (!isMeasuring) {
                startOri = {...currentOri};
                isMeasuring = true;
                mainBtn.innerText = "끝점 & 분석시작";
                mainBtn.style.background = "#007AFF";
            } else {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCanvas.getContext('2d').drawImage(video, 0, 0);
                const base64Img = tempCanvas.toDataURL('image/jpeg', 0.8);

                sendMessage({
                    img: base64Img, 
                    dist: finalDist,
                    ts: Date.now()
                });
                
                status.innerText = "데이터 전송 중...";
                mainBtn.innerText = "분석 중...";
                mainBtn.disabled = true;
            }
        };

        resetBtn.onclick = () => { window.location.reload(); };
    </script>
    """
    return components.html(ar_html, height=550)

# 5. 메인 실행부 (에러 방어 로직 강화)
# 컴포넌트 결과값을 안전하게 가져오기
result_data = ar_scanner_component()

# 에러가 발생했던 부분: result_data가 None이거나 dict가 아닐 경우를 철저히 체크
if result_data and isinstance(result_data, dict):
    # .get()을 사용하여 키가 없어도 에러가 나지 않게 함
    captured_img_b64 = result_data.get("img")
    measured_dist = result_data.get("dist", 0)

    if captured_img_b64 and measured_dist > 0:
        st.divider()
        st.subheader("🔍 분석 리포트")
        
        try:
            # Base64 이미지 디코딩 및 전처리
            img_data = base64.b64decode(captured_img_b64.split(',')[1])
            image = Image.open(BytesIO(img_data))
            img_array = np.array(image.convert("RGB"))

            with st.spinner("AI 균열 정밀 분석 중..."):
                res = model_crack.predict(source=img_array, conf=0.25, verbose=False)
                final_img, area, thickness = analyze_captured_image(img_array, measured_dist, res)
                
                st.image(final_img, caption=f"측정 거리 {measured_dist:.3f}m 기준 분석 결과", use_container_width=True)
                
                c1, c2 = st.columns(2)
                c1.metric("📐 탐지된 균열 면적", f"{area:.2f} cm²")
                c2.metric("🔥 최대 균열 폭", f"{thickness:.2f} mm")
        except Exception as e:
            st.error(f"이미지 분석 중 오류가 발생했습니다: {e}")
    else:
        st.info("준비되었습니다. 위 화면에서 균열의 시작과 끝을 지정해주세요.")
else:
    # 최초 로드 시 result_data는 None이므로 아무것도 하지 않고 대기
    st.info("카메라 권한을 허용하고 [시작점 고정]을 눌러주세요.")
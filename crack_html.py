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
st.title("🏗️ AR 실시간 균열 정밀 진단 V5.3")

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
    
    # 세로 모드 FOV 보정 (아이폰 세로 기준 가로 화각은 약 50-60도)
    fov_radians = math.radians(60 / 2)
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

# 4. AR 카메라 컴포넌트 (세로 모드 최적화)
def ar_scanner_component():
    ar_html = """
    <div id="wrapper" style="position: relative; width: 100%; height: 600px; background: #000; border-radius: 20px; overflow: hidden; border: 3px solid #333;">
        <video id="video" style="width: 100%; height: 100%; object-fit: cover;" autoplay playsinline></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
        
        <div id="ui" style="position: absolute; bottom: 30px; width: 100%; display: flex; justify-content: center; gap: 15px;">
            <button id="mainBtn" style="padding: 18px 28px; background: #FF4B4B; color: white; border: none; border-radius: 40px; font-weight: bold; font-size: 18px; min-width: 160px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">시작점 고정</button>
            <button id="resetBtn" style="padding: 18px 28px; background: #444; color: white; border: none; border-radius: 40px;">리셋</button>
        </div>
        <div id="status" style="position: absolute; top: 20px; left: 20px; color: #00FF00; background: rgba(0,0,0,0.8); padding: 10px 18px; border-radius: 10px; font-size: 14px; font-family: monospace;">PORTRAIT MODE READY</div>
    </div>

    <script>
        function sendMessage(data) {
            if (window.Streamlit) {
                window.Streamlit.setComponentValue(data);
            }
        }

        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const mainBtn = document.getElementById('mainBtn');
        const status = document.getElementById('status');

        let isMeasuring = false;
        let startOri = null;
        let currentOri = { alpha: 0, beta: 0, gamma: 0 };
        let finalDist = 0;

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(s => { video.srcObject = s; });

        async function initSensor() {
            if (window.DeviceOrientationEvent && typeof DeviceOrientationEvent.requestPermission === 'function') {
                const res = await DeviceOrientationEvent.requestPermission();
                if (res === 'granted') {
                    window.addEventListener('deviceorientation', e => { 
                        currentOri = {alpha: e.alpha, beta: e.beta, gamma: e.gamma}; 
                    });
                }
            } else {
                window.addEventListener('deviceorientation', e => { 
                    currentOri = {alpha: e.alpha || 0, beta: e.beta || 0, gamma: e.gamma || 0}; 
                });
            }
        }

        function draw() {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;

            // 조준점
            ctx.strokeStyle = "white"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(centerX, centerY, 20, 0, Math.PI*2); ctx.stroke();
            ctx.fillStyle = "red"; ctx.beginPath(); ctx.arc(centerX, centerY, 4, 0, Math.PI*2); ctx.fill();

            if (isMeasuring && startOri) {
                // 세로 모드(Portrait) 각도 보정 로직
                // 좌우 이동: gamma(혹은 alpha), 상하 이동: beta
                const sensitivity = 35; 
                const dx = (currentOri.gamma - startOri.gamma) * sensitivity;
                const dy = (currentOri.beta - startOri.beta) * sensitivity;
                
                const startX = centerX - dx;
                const startY = centerY + dy;

                ctx.fillStyle = "#00FF00";
                ctx.beginPath(); ctx.arc(startX, startY, 10, 0, Math.PI*2); ctx.fill();

                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = "#00FF00";
                ctx.lineWidth = 3;
                ctx.beginPath(); ctx.moveTo(startX, startY); ctx.lineTo(centerX, centerY); ctx.stroke();
                ctx.setLineDash([]);

                const dG = Math.abs(currentOri.gamma - startOri.gamma) * (Math.PI/180);
                const dB = Math.abs(currentOri.beta - startOri.beta) * (Math.PI/180);
                finalDist = Math.sqrt(Math.pow(dG, 2) + Math.pow(dB, 2)) * 1.2; // 세로 보정 계수
                status.innerText = "MEASURING: " + finalDist.toFixed(3) + "m";
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
                // 캡처 전송
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCanvas.getContext('2d').drawImage(video, 0, 0);
                
                // 데이터 크기를 줄이기 위해 압축률 조정 (0.7)
                const base64Img = tempCanvas.toDataURL('image/jpeg', 0.7);

                sendMessage({
                    img: base64Img, 
                    dist: finalDist,
                    ts: Date.now()
                });
                
                status.innerText = "전송 완료! 분석을 기다리세요...";
                mainBtn.innerText = "분석 중...";
                mainBtn.disabled = true;
                mainBtn.style.background = "#888";
            }
        };

        document.getElementById('resetBtn').onclick = () => { window.location.reload(); };
    </script>
    """
    return components.html(ar_html, height=650)

# 5. 실행부
result_data = ar_scanner_component()

if result_data and isinstance(result_data, dict):
    captured_img_b64 = result_data.get("img")
    measured_dist = result_data.get("dist", 0)

    if captured_img_b64 and measured_dist > 0:
        st.divider()
        with st.status("🚀 AI 분석 진행 중...", expanded=True) as status:
            try:
                img_data = base64.b64decode(captured_img_b64.split(',')[1])
                image = Image.open(BytesIO(img_data))
                img_array = np.array(image.convert("RGB"))

                res = model_crack.predict(source=img_array, conf=0.25, verbose=False)
                final_img, area, thickness = analyze_captured_image(img_array, measured_dist, res)
                
                st.image(final_img, caption="AI 균열 진단 결과", use_container_width=True)
                
                m1, m2 = st.columns(2)
                m1.metric("📐 균열 총 면적", f"{area:.2f} cm²")
                m2.metric("🔥 최대 균열 폭", f"{thickness:.2f} mm")
                
                status.update(label="✅ 분석 완료!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"분석 중 오류: {e}")
    else:
        st.info("세로 모드로 균열을 가로지르며 측정해 주세요.")
else:
    st.info("카메라가 준비되면 [시작점 고정] 버튼을 누르세요.")
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
st.title("🏗️ AR 실시간 균열 정밀 진단 V5")

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

# 3. 분석 함수 (물리적 거리 기반 스케일링 적용)
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
    
    # 두 점 사이의 거리가 화면의 가로 폭 전체에 해당한다고 가정하는 대신, 
    # FOV를 이용해 1픽셀당 실제 길이를 구함 (아이폰 메인 카메라 FOV 약 70도)
    fov_radians = math.radians(70 / 2)
    real_view_width_cm = 2 * (physical_dist_m * 100) * math.tan(fov_radians)
    cm_per_px = real_view_width_cm / image_np.shape[1]
    
    # 마스크 오버레이 (빨간색)
    draw_img[mask_canvas == 1] = [255, 0, 0]
    
    # 두께 계산
    max_thickness_px = 0
    if np.sum(mask_canvas) > 0:
        dist_transform = cv2.distanceTransform(mask_canvas, cv2.DIST_L2, 3)
        max_thickness_px = np.max(dist_transform) * 2
        
    area_cm2 = np.sum(mask_canvas) * (cm_per_px**2)
    thick_mm = max_thickness_px * (cm_per_px * 10)
    
    return draw_img, area_cm2, thick_mm

# 4. 고성능 AR 카메라 & 자동 캡처 컴포넌트
def ar_scanner_component():
    ar_html = """
    <div id="wrapper" style="position: relative; width: 100%; height: 550px; background: #000; border-radius: 20px; overflow: hidden;">
        <video id="video" style="width: 100%; height: 100%; object-fit: cover;" autoplay playsinline></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
        
        <div id="ui" style="position: absolute; bottom: 30px; width: 100%; display: flex; justify-content: center; gap: 20px;">
            <button id="mainBtn" style="padding: 15px 30px; background: #FF4B4B; color: white; border: none; border-radius: 50px; font-weight: bold; font-size: 18px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">시작점 고정</button>
            <button id="resetBtn" style="padding: 15px 30px; background: #444; color: white; border: none; border-radius: 50px;">초기화</button>
        </div>
        <div id="status" style="position: absolute; top: 20px; left: 20px; color: #00FF00; background: rgba(0,0,0,0.6); padding: 10px 20px; border-radius: 10px; font-family: monospace;">AR SYSTEM READY</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const mainBtn = document.getElementById('mainBtn');
        const resetBtn = document.getElementById('resetBtn');
        const status = document.getElementById('status');

        let isMeasuring = false;
        let startOri = null;
        let currentOri = { alpha: 0, beta: 0 };

        // 1. 카메라 기동
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: 1280, height: 720 } })
            .then(s => { video.srcObject = s; });

        // 2. 센서 리스너
        async function getPerm() {
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                const res = await DeviceOrientationEvent.requestPermission();
                if (res === 'granted') window.addEventListener('deviceorientation', e => { currentOri = {alpha: e.alpha, beta: e.beta}; });
            } else {
                window.addEventListener('deviceorientation', e => { currentOri = {alpha: e.alpha, beta: e.beta}; });
            }
        }

        // 3. 루프 그리기 (시각적 선 표시)
        function draw() {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 중앙 레티클
            ctx.strokeStyle = "white"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(canvas.width/2, canvas.height/2, 20, 0, Math.PI*2); ctx.stroke();
            ctx.fillStyle = "red"; ctx.beginPath(); ctx.arc(canvas.width/2, canvas.height/2, 3, 0, Math.PI*2); ctx.fill();

            if (isMeasuring && startOri) {
                // 시작점(화면 중앙 가상 고정)과 현재점 연결
                ctx.setLineDash([5, 5]);
                ctx.strokeStyle = "#00FF00";
                ctx.beginPath();
                ctx.moveTo(canvas.width/2, canvas.height/2);
                // 실제로 공간에 고정된 점을 시뮬레이션하기 위해 반대 방향으로 선을 그음
                const dx = (currentOri.alpha - startOri.alpha) * 10;
                const dy = (currentOri.beta - startOri.beta) * 10;
                ctx.lineTo(canvas.width/2 - dx, canvas.height/2 + dy);
                ctx.stroke();
                
                const dAlpha = Math.abs(currentOri.alpha - startOri.alpha) * (Math.PI/180);
                const dBeta = Math.abs(currentOri.beta - startOri.beta) * (Math.PI/180);
                const dist = Math.sqrt(Math.pow(dAlpha, 2) + Math.pow(dBeta, 2)) * 1.5;
                status.innerText = "MEASURING: " + dist.toFixed(3) + "m";
            }
            requestAnimationFrame(draw);
        }
        draw();

        mainBtn.onclick = async () => {
            await getPerm();
            if (!isMeasuring) {
                startOri = {...currentOri};
                isMeasuring = true;
                mainBtn.innerText = "끝점 & 캡처";
                mainBtn.style.background = "#007AFF";
            } else {
                const dAlpha = Math.abs(currentOri.alpha - startOri.alpha) * (Math.PI/180);
                const dBeta = Math.abs(currentOri.beta - startOri.beta) * (Math.PI/180);
                const dist = Math.sqrt(Math.pow(dAlpha, 2) + Math.pow(dBeta, 2)) * 1.5;
                
                // 사진 캡처
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCanvas.getContext('2d').drawImage(video, 0, 0);
                const base64Img = tempCanvas.toDataURL('image/jpeg');

                // 데이터 전송
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue', 
                    value: {img: base64Img, dist: dist}
                }, '*');
                
                isMeasuring = false;
                mainBtn.innerText = "분석 완료";
                mainBtn.disabled = true;
            }
        };

        resetBtn.onclick = () => { location.reload(); };
    </script>
    """
    return components.html(ar_html, height=600)

# 5. 메인 로직
ar_data = ar_scanner_component()

if ar_data is not None and isinstance(ar_data, dict):
    captured_img_b64 = ar_data.get("img")
    measured_dist = ar_data.get("dist")

    if captured_img_b64:
        # Base64를 이미지로 변환
        img_data = base64.b64decode(captured_img_b64.split(',')[1])
        image = Image.open(BytesIO(img_data))
        img_array = np.array(image.convert("RGB"))

        st.success(f"✅ 캡처 완료! 측정된 균열 거리: {measured_dist:.3f}m")
        
        with st.spinner("AI가 균열 면적을 계산하고 있습니다..."):
            res = model_crack.predict(source=img_array, conf=0.25, verbose=False)
            final_img, area, thickness = analyze_captured_image(img_array, measured_dist, res)
            
            st.image(final_img, caption="AI 진단 결과", use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("📐 균열 총 면적", f"{area:.2f} cm²")
            c2.metric("🔥 최대 균열 폭", f"{thickness:.2f} mm")
            
            st.info("💡 면적은 AR로 측정한 두 지점 사이의 거리를 기준으로 사진 전체 스케일을 보정하여 계산되었습니다.")

else:
    st.write("👆 위 화면에서 **시작점**을 찍고 균열 끝으로 이동한 뒤 **캡처** 버튼을 누르세요.")
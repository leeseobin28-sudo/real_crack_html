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

# 1. 초기 설정
st.set_page_config(page_title="벽면 균열 실시간 진단", page_icon="🏗️", layout="wide")

# 모델 로드 (캐싱)
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

# 2. 분석 함수
def analyze_captured_image(image_np, physical_dist_m, results_crack):
    draw_img = image_np.copy()
    if not results_crack or len(results_crack) == 0 or results_crack[0].masks is None:
        return draw_img, 0, 0
    
    result = results_crack[0]
    mask_canvas = np.zeros(image_np.shape[:2], dtype=np.uint8)
    
    for mask in result.masks.xy:
        if len(mask) > 0:
            pts = np.array(mask, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_canvas, [pts], 1)
    
    # 세로 모드(Portrait) 화각 보정 (아이폰 기준 약 58도)
    fov_radians = math.radians(58 / 2)
    real_view_width_cm = 2 * (physical_dist_m * 100) * math.tan(fov_radians)
    cm_per_px = real_view_width_cm / image_np.shape[1]
    
    # 결과 시각화
    draw_img[mask_canvas == 1] = [255, 0, 0] # 균열 빨간색
    dist_transform = cv2.distanceTransform(mask_canvas, cv2.DIST_L2, 3)
    max_thickness_px = np.max(dist_transform) * 2 if np.any(dist_transform) else 0
    
    area_cm2 = np.sum(mask_canvas) * (cm_per_px**2)
    thick_mm = max_thickness_px * (cm_per_px * 10)
    
    return draw_img, area_cm2, thick_mm

# 3. 벽면 특화 AR 컴포넌트
def ar_wall_scanner():
    ar_html = """
    <div id="wrapper" style="position: relative; width: 100%; height: 550px; background: #000; border-radius: 15px; overflow: hidden;">
        <video id="video" style="width: 100%; height: 100%; object-fit: cover;" autoplay playsinline></video>
        <canvas id="overlay" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
        
        <div id="controls" style="position: absolute; bottom: 20px; width: 100%; display: flex; justify-content: center; gap: 10px;">
            <button id="btnAction" style="padding: 15px 25px; background: #FF4B4B; color: white; border: none; border-radius: 30px; font-weight: bold;">시작점 고정</button>
            <button id="btnReset" style="padding: 15px 25px; background: #333; color: white; border: none; border-radius: 30px;">초기화</button>
        </div>
        <div id="info" style="position: absolute; top: 15px; left: 15px; color: #00FF00; background: rgba(0,0,0,0.7); padding: 5px 12px; border-radius: 5px; font-family: monospace; font-size: 12px;">WALL SCANNER READY</div>
    </div>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        const btnAction = document.getElementById('btnAction');
        const info = document.getElementById('info');

        let isMeasuring = false;
        let startPoint = null;
        let currentOri = { a: 0, b: 0, g: 0 };
        let distResult = 0;

        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } }).then(s => video.srcObject = s);

        async function requestPerm() {
            if (typeof DeviceOrientationEvent.requestPermission === 'function') {
                const res = await DeviceOrientationEvent.requestPermission();
                if (res === 'granted') window.addEventListener('deviceorientation', e => { currentOri = {a:e.alpha, b:e.beta, g:e.gamma}; });
            } else {
                window.addEventListener('deviceorientation', e => { currentOri = {a:e.alpha, b:e.beta, g:e.gamma}; });
            }
        }

        function draw() {
            canvas.width = video.clientWidth;
            canvas.height = video.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const cx = canvas.width / 2;
            const cy = canvas.height / 2;

            // 중앙 레티클
            ctx.strokeStyle = "white"; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(cx, cy, 20, 0, Math.PI*2); ctx.stroke();
            ctx.fillStyle = "red"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI*2); ctx.fill();

            if (isMeasuring && startPoint) {
                // 벽면(90도 세움) 기준 좌표 보정: alpha(좌우), beta(상하) 사용
                // 폰을 세웠을 때 alpha의 변화가 좌우 이동을 가장 잘 나타냄
                let dx = (currentOri.a - startPoint.a);
                if (dx > 180) dx -= 360; if (dx < -180) dx += 360; // 각도 끊김 방지
                
                const dy = (currentOri.b - startPoint.b);

                const sens = 30; // 화면 표시 감도
                const sx = cx - (dx * sens);
                const sy = cy + (dy * sens);

                // 시작점 표시 (초록점)
                ctx.fillStyle = "#00FF00";
                ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI*2); ctx.fill();

                // 연결 점선
                ctx.setLineDash([5, 5]);
                ctx.strokeStyle = "#00FF00";
                ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(cx, cy); ctx.stroke();
                ctx.setLineDash([]);

                // 실제 물리 거리 계산 (라디안 변환)
                const radX = Math.abs(dx) * (Math.PI/180);
                const radY = Math.abs(dy) * (Math.PI/180);
                distResult = Math.sqrt(Math.pow(radX, 2) + Math.pow(radY, 2)) * 1.5; // 벽면 보정계수
                info.innerText = "MEASURING: " + distResult.toFixed(3) + "m";
            }
            requestAnimationFrame(draw);
        }
        draw();

        btnAction.onclick = async () => {
            await requestPerm();
            if (!isMeasuring) {
                startPoint = {...currentOri};
                isMeasuring = true;
                btnAction.innerText = "분석 시작 (캡처)";
                btnAction.style.background = "#007AFF";
            } else {
                // 캡처 후 데이터 전송
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = video.videoWidth;
                tempCanvas.height = video.videoHeight;
                tempCanvas.getContext('2d').drawImage(video, 0, 0);
                const b64 = tempCanvas.toDataURL('image/jpeg', 0.7);

                if (window.Streamlit) {
                    window.Streamlit.setComponentValue({ img: b64, dist: distResult, ts: Date.now() });
                }
                btnAction.innerText = "분석 중...";
                btnAction.disabled = true;
            }
        };
        document.getElementById('btnReset').onclick = () => window.location.reload();
    </script>
    """
    return components.html(ar_html, height=580)

# 4. 메인 실행 로직
st.subheader("🏗️ 벽면 전용 실시간 균열 분석기")

# 결과값을 세션 스테이트로 관리하여 새로고침 시 날아가는 것 방지
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

res_data = ar_wall_scanner()

# 데이터가 들어오면 분석 실행
if res_data and isinstance(res_data, dict) and "img" in res_data:
    b64_img = res_data.get("img")
    dist = res_data.get("dist", 0)
    
    if b64_img and dist > 0:
        st.divider()
        with st.status("🔍 AI 균열 분석 엔진 가동 중...", expanded=True) as status:
            try:
                # 1. 이미지 복원
                raw_b64 = b64_img.split(',')[1]
                img_bytes = base64.b64decode(raw_b64)
                img = Image.open(BytesIO(img_bytes))
                img_np = np.array(img.convert("RGB"))

                # 2. YOLO 예측
                results = model_crack.predict(source=img_np, conf=0.25, verbose=False)
                
                # 3. 수치 계산 및 그리기
                processed_img, area, thick = analyze_captured_image(img_np, dist, results)

                # 4. 결과 출력
                st.image(processed_img, caption=f"분석 완료 (측정 거리: {dist:.3f}m)", use_container_width=True)
                
                c1, c2 = st.columns(2)
                c1.metric("📐 균열 총 면적", f"{area:.2f} cm²")
                c2.metric("🔥 최대 균열 폭", f"{thick:.2f} mm")
                
                status.update(label="✅ 분석이 성공적으로 완료되었습니다!", state="complete", expanded=False)
                st.session_state.analysis_done = True
                
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")
                status.update(label="❌ 분석 실패", state="error")

else:
    st.info("💡 **사용 방법**\n1. 폰을 벽과 평행하게 **90도로 세웁니다.**\n2. 균열 시작점에 빨간점을 맞추고 **[시작점 고정]**을 누릅니다.\n3. 폰을 옆으로 천천히 움직여 균열 끝에서 **[분석 시작]**을 누릅니다.")
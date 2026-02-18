import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np


st.set_page_config(
    page_title="Hand Gesture Recognition System",
    layout="wide"
)


st.markdown("""
<style>
.block-container { padding-top: 0rem !important; }
.stApp { background-color: #f4f6f9; }

.header {
    background-color: #1976d2;
    padding: 40px 45px;
    border-radius: 0px 0px 14px 14px;
    margin-bottom: 30px;
}

.header h1 {
    margin: 0;
    font-size: 44px;
    font-weight: 800;
    color: white;
}

.header p {
    margin-top: 14px;
    font-size: 16px;
    font-weight: 500;
    color: black;
    background-color: rgba(255,255,255,0.9);
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
}

.stButton > button {
    background-color: #1976d2;
    color: white;
    border-radius: 6px;
    padding: 0.65rem 1.7rem;
    font-size: 15px;
    font-weight: 600;
    border: none;
}

.status-box {
    background-color: white;
    padding: 14px;
    border-radius: 8px;
    border-left: 6px solid #1976d2;
    margin-bottom: 12px;
    color: black;
}

.info-box {
    background-color: white;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #ddd;
    margin-bottom: 8px;
    font-weight: 600;
    color: black;
}

h3, label {
    color: black !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

st.markdown("""
<div class="header">
    <h1>Hand Gesture Recognition System</h1>
    <p>Real-time hand detection using OpenCV and MediaPipe Hands</p>
</div>
""", unsafe_allow_html=True)


spacer, b1, b2, b3 = st.columns([6, 1, 1, 1])

with b1:
    if st.button("▶ Start"):
        st.session_state.camera_on = True

with b2:
    if st.button("■ Stop"):
        st.session_state.camera_on = False

with b3:
    st.button("📸 Capture")


left_col, cam_col, right_col = st.columns([1.3, 3.2, 1.3])


with left_col:
    st.markdown("### Detection Status")
    status_box = st.empty()

    st.markdown("### Detection Info")
    info_box = st.empty()


camera_placeholder = cam_col.empty()

with right_col:
    st.markdown("### Detection Parameters")
    det_conf = st.slider("Detection Confidence", 0.0, 1.0, 0.75)
    track_conf = st.slider("Tracking Confidence", 0.0, 1.0, 0.80)
    max_hands = st.slider("Max Number of Hands", 1, 4, 2)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
cap = None
prev_time = 0


while True:
    if st.session_state.camera_on:
        if cap is None:
            cap = cv2.VideoCapture(0)

        with mp_hands.Hands(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            max_num_hands=max_hands
        ) as hands:

            success, frame = cap.read()
            if not success:
                st.error("Camera not accessible")
                break

        
            frame = cv2.flip(frame, 1)

 
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_count = 0
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    hand_count += 1

         
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if prev_time != 0 else 0
            prev_time = curr_time

            camera_placeholder.image(
                frame,
                channels="BGR",
                width='stretch'
            )

            status_box.markdown(f"""
            <div class="status-box">
                <b>Camera:</b> Active<br>
                <b>Hands:</b> {hand_count}<br>
                <b>FPS:</b> {fps}<br>
                <b>Model:</b> MediaPipe Hands
            </div>
            """, unsafe_allow_html=True)

            info_box.markdown(f"""
            <div class="info-box">{21 * hand_count}<br>Landmarks</div>
            <div class="info-box">{15 * hand_count}<br>Connections</div>
            <div class="info-box">640×480<br>Resolution</div>
            <div class="info-box">{int(1000/fps) if fps>0 else 0} ms<br>Latency</div>
            """, unsafe_allow_html=True)
            print(f"[Camera: Active | Hands: {hand_count} | FPS: {fps} | Landmarks: {21 * hand_count} | Connections: {15 * hand_count} | Latency: {int(1000/fps) if fps>0 else 0}ms]")

    else:
        if cap:
            cap.release()
            cap = None

        status_box.markdown("""
        <div class="status-box">
            <b>Camera:</b> Inactive<br>
            <b>Hands:</b> 0<br>
            <b>FPS:</b> 0<br>
            <b>Model:</b> Not Running
        </div>
        """, unsafe_allow_html=True)

        info_box.markdown("""
        <div class="info-box">0<br>Landmarks</div>
        <div class="info-box">0<br>Connections</div>
        <div class="info-box">—<br>Resolution</div>
        <div class="info-box">0 ms<br>Latency</div>
        """, unsafe_allow_html=True)

        # Print to terminal
        print("[Camera: Inactive | Hands: 0 | FPS: 0 | Landmarks: 0 | Connections: 0]")

        camera_placeholder.info("Camera is stopped")
        time.sleep(0.2)

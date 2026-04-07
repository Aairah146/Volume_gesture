import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np
import math
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


def set_system_volume(percent: int):
    
    if not AUDIO_AVAILABLE:
        return
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
        scalar = max(0.0, min(1.0, percent / 100.0))
        volume_ctrl.SetMasterVolumeLevelScalar(scalar, None)
    except Exception:
        pass


def smooth_volume(current_smoothed: float, new_raw: int, alpha: float = 0.2) -> float:
    return alpha * new_raw + (1 - alpha) * current_smoothed

def calculate_distance(point1, point2, frame_width, frame_height):

    x1, y1 = int(point1.x * frame_width), int(point1.y * frame_height)
    x2, y2 = int(point2.x * frame_width), int(point2.y * frame_height)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance, (x1, y1), (x2, y2)


def get_finger_states(hand_landmarks, frame_width, frame_height):
    
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    
    wrist_x, wrist_y = int(wrist.x * frame_width), int(wrist.y * frame_height)
    mcp_x, mcp_y = int(middle_mcp.x * frame_width), int(middle_mcp.y * frame_height)
    
    
    base_ref_dist = math.sqrt((mcp_x - wrist_x)**2 + (mcp_y - wrist_y)**2)
    
    
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]  
    
    extended_count = 0
    tip_distances = []
    
    for tip_id, pip_id in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        
        tip_x, tip_y = int(tip.x * frame_width), int(tip.y * frame_height)
        pip_x, pip_y = int(pip.x * frame_width), int(pip.y * frame_height)
        
       
        tip_dist = math.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2)
        pip_dist = math.sqrt((pip_x - wrist_x)**2 + (pip_y - wrist_y)**2)
        
        tip_distances.append(tip_dist)
        
        
        if tip_dist > pip_dist * 1.15 and tip_dist > base_ref_dist * 0.8:
            extended_count += 1
    
    avg_tip_distance = sum(tip_distances) / len(tip_distances)
    return extended_count, avg_tip_distance


def classify_gesture(distance, hand_landmarks, frame_width, frame_height):

    extended_count, avg_tip_distance = get_finger_states(hand_landmarks, frame_width, frame_height)

    # --- Strong rule: a closed fist must always be Mute ---
    # This prevents a fist (thumb/index close) from being misread as "Pinch 0%".
    if extended_count == 0:
        return "Closed Hand - Mute", (0, 0, 255), "mute"

    # --- Compute whether index finger is extended (needed for pinch) ---
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]

    wrist_x, wrist_y = int(wrist.x * frame_width), int(wrist.y * frame_height)
    mcp_x, mcp_y = int(middle_mcp.x * frame_width), int(middle_mcp.y * frame_height)
    base_ref_dist = math.sqrt((mcp_x - wrist_x) ** 2 + (mcp_y - wrist_y) ** 2)

    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    itx, ity = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
    ipx, ipy = int(index_pip.x * frame_width), int(index_pip.y * frame_height)
    index_tip_dist = math.sqrt((itx - wrist_x) ** 2 + (ity - wrist_y) ** 2)
    index_pip_dist = math.sqrt((ipx - wrist_x) ** 2 + (ipy - wrist_y) ** 2)

    index_extended = (index_tip_dist > index_pip_dist * 1.12) and (index_tip_dist > base_ref_dist * 0.8)

    # Open hand should win over pinch (otherwise an open palm with thumb/index closer can be misread as pinch).
    if extended_count >= 3:
        return "Open Hand - Unmute", (0, 255, 0), "unmute"

    # Pinch is only valid if the index finger is extended (avoids fist false-positives).
    # Return a generic pinch label; the UI will show the *actual* computed volume so it never disagrees.
    if index_extended and extended_count <= 2 and distance < 170:
        return "Pinch", (255, 0, 255), "volume"

    return "Hand Detected", (255, 255, 255), "none"


def map_distance_to_volume(distance, min_dist=20, max_dist=200):
    """Map distance to volume level (0-100%)."""
    # Clamp distance within range
    distance = np.clip(distance, min_dist, max_dist)
    # Normalize
    normalized = (distance - min_dist) / (max_dist - min_dist)
    # Scale to 0–100
    volume = int(normalized * 100)
    return volume


def apply_beauty_blur(
    frame,
    strength: float = 0.35,
    brightness: int = 12,
    contrast: float = 1.08,
    gamma: float = 0.92,
    saturation: float = 1.06,
):
    """Beauty filter: soft blur + bright/glow tone (fast).

    Applied BEFORE drawing overlays so landmarks/text stay crisp.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    contrast = float(max(0.1, contrast))
    gamma = float(max(0.05, gamma))
    saturation = float(max(0.0, saturation))

    out = frame

    # Soft smoothing
    if strength > 0.0:
        sigma = 1.2 + 2.8 * strength
        smoothed = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha = 0.22 + 0.62 * strength
        out = cv2.addWeighted(smoothed, alpha, out, 1.0 - alpha, 0)

    # Brightness/contrast (glow-like exposure)
    if brightness != 0 or contrast != 1.0:
        out = cv2.convertScaleAbs(out, alpha=contrast, beta=int(brightness))

    # Gamma (gamma < 1 => brighter)
    if abs(gamma - 1.0) > 1e-3:
        out = cv2.LUT(out, _get_gamma_lut(gamma))

    # Slight saturation boost
    if abs(saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return out


@lru_cache(maxsize=64)
def _get_gamma_lut(gamma: float):
    g = float(round(gamma, 3))
    inv_gamma = 1.0 / g
    return np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")


def draw_gesture_info(frame, distance, gesture_name, volume, thumb_pos, index_pos, color):
    """Draw gesture information and visual elements on frame."""
    fh, fw = frame.shape[:2]

    # Draw line between thumb and index finger
    cv2.line(frame, thumb_pos, index_pos, color, 3)
    
    # Draw circles at thumb and index positions
    cv2.circle(frame, thumb_pos, 6, (255, 0, 255), cv2.FILLED)
    cv2.circle(frame, index_pos, 6, (255, 0, 255), cv2.FILLED)
    
    # Draw midpoint circle
    mid_x = (thumb_pos[0] + index_pos[0]) // 2
    mid_y = (thumb_pos[1] + index_pos[1]) // 2
    cv2.circle(frame, (mid_x, mid_y), 5, color, cv2.FILLED)
    
    # Display distance value near midpoint (smaller)
    mid_font_scale = 0.6 if fw >= 480 else 0.55
    cv2.putText(
        frame,
        f"{int(distance)}px",
        (mid_x - 26, mid_y - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        mid_font_scale,
        (255, 255, 255),
        2,
    )
    
    # Display gesture information overlay (much smaller so it doesn't cover face)
    overlay_y = 30
    overlay_w = min(max(210, int(fw * 0.52)), 360)
    overlay_h = 72
    cv2.rectangle(frame, (10, 10), (10 + overlay_w, 10 + overlay_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (10 + overlay_w, 10 + overlay_h), color, 2)
    
    title_scale = 0.56 if fw >= 480 else 0.52
    line_scale = 0.50 if fw >= 480 else 0.46
    cv2.putText(
        frame,
        f"Gest: {gesture_name}",
        (18, overlay_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"Dist: {int(distance)} px",
        (18, overlay_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        line_scale,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Vol: {volume}%",
        (18, overlay_y + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        line_scale,
        (255, 255, 255),
        2,
    )
    
    return frame


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
    padding: 22px 26px;
    border-radius: 0px 0px 14px 14px;
    margin-bottom: 14px;
}

.header h1 {
    margin: 0;
    font-size: 34px;
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
    padding: 10px 12px;
    border-radius: 8px;
    border-left: 6px solid #1976d2;
    margin-bottom: 10px;
    color: black;
}

.info-box {
    background-color: white;
    padding: 8px 10px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #ddd;
    margin-bottom: 0px;
    font-weight: 600;
    color: black;
}

.info-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.info-grid .info-box {
    flex: 1 1 calc(50% - 10px);
    min-width: 130px;
}

.m3-header {
    background: linear-gradient(135deg, #00897b, #00695c);
    padding: 18px 24px;
    border-radius: 10px;
    margin: 20px 0 16px 0;
}

.m3-header h3 {
    color: white !important;
    margin: 0;
    font-size: 22px;
    font-weight: 800;
}

.m3-header p {
    color: rgba(255,255,255,0.88);
    margin: 6px 0 0;
    font-size: 13px;
}

.volume-card {
    background: white;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.volume-number {
    font-size: 64px;
    font-weight: 800;
    color: #00897b;
    line-height: 1;
}

.volume-unit {
    font-size: 18px;
    color: #888;
    margin-top: 4px;
}

.badge-active {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 700;
    margin: 10px 4px 0;
}

.badge-synced {
    display: inline-block;
    background: #e3f2fd;
    color: #1565c0;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 700;
    margin: 10px 4px 0;
}

.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #333;
    margin: 10px 0 8px;
}

.camera-wrap {
    background: white;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
    padding: 10px;
}

.tight-hr {
    margin: 14px 0 10px;
    border: none;
    border-top: 1px solid #ddd;
}

h3, label {
    color: black !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "volume_history" not in st.session_state:
    st.session_state.volume_history = []
if "current_volume" not in st.session_state:
    st.session_state.current_volume = 0
if "smoothed_volume" not in st.session_state:
    st.session_state.smoothed_volume = 0.0
if "last_nonzero_volume" not in st.session_state:
    st.session_state.last_nonzero_volume = 50
if "last_applied_volume" not in st.session_state:
    st.session_state.last_applied_volume = None
if "last_apply_ts" not in st.session_state:
    st.session_state.last_apply_ts = 0.0
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "last_chart_update" not in st.session_state:
    st.session_state.last_chart_update = 0.0
if "hands" not in st.session_state:
    st.session_state.hands = None
if "hands_cfg" not in st.session_state:
    st.session_state.hands_cfg = None
if "last_gesture_name" not in st.session_state:
    st.session_state.last_gesture_name = "No gesture detected"
if "last_gesture_type" not in st.session_state:
    st.session_state.last_gesture_type = "none"
if "last_gesture_ts" not in st.session_state:
    st.session_state.last_gesture_ts = 0.0
if "cap" not in st.session_state:
    st.session_state.cap = None
if "cap_backend" not in st.session_state:
    st.session_state.cap_backend = None
if "last_log_ts" not in st.session_state:
    st.session_state.last_log_ts = 0.0
if "prev_time" not in st.session_state:
    st.session_state.prev_time = 0.0

st.markdown("""
<div class="header">
    <h1>Hand Gesture Recognition System</h1>
    <p>Real-time hand detection using OpenCV and MediaPipe Hands</p>
</div>
""", unsafe_allow_html=True)

left_spacer, b1, b2, b3, right_spacer = st.columns([2, 1, 1, 1, 2])

with b1:
    if st.button("▶ Start"):
        st.session_state.camera_on = True

with b2:
    if st.button("■ Stop"):
        st.session_state.camera_on = False

with b3:
    st.button("📸 Capture")


# ===== MAIN AREA: Left (small cards) | Center (camera focus) | Right (charts) =====
left_col, center_col, right_col = st.columns([1, 2.2, 1])

with left_col:
    st.markdown('<div class="section-title">🧭 Status</div>', unsafe_allow_html=True)
    status_box = st.empty()

    st.markdown('<div class="section-title">ℹ️ Detection Info</div>', unsafe_allow_html=True)
    info_box = st.empty()

    st.markdown('<div class="section-title">✋ Gesture</div>', unsafe_allow_html=True)
    gesture_info = st.empty()

with center_col:
    st.markdown('<div class="section-title">📹 Live Camera</div>', unsafe_allow_html=True)
    st.markdown('<div class="camera-wrap">', unsafe_allow_html=True)
    camera_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">🔊 Current Volume</div>', unsafe_allow_html=True)
    current_vol_placeholder = st.empty()
    st.markdown('<div class="section-title">📈 Distance → Volume</div>', unsafe_allow_html=True)
    mapping_chart_placeholder = st.empty()
    st.markdown('<div class="section-title">🕘 Volume History</div>', unsafe_allow_html=True)
    history_chart_placeholder = st.empty()

    st.markdown('<div class="section-title">🔧 Detection Parameters</div>', unsafe_allow_html=True)
    det_conf = st.slider("Detection Confidence", 0.0, 1.0, 0.75, key="det_conf")
    track_conf = st.slider("Tracking Confidence", 0.0, 1.0, 0.80, key="track_conf")
    max_hands = st.slider("Max Number of Hands", 1, 4, 2, key="max_hands")

st.markdown("<hr class='tight-hr'>", unsafe_allow_html=True)


# ===== Helper: render charts =====
def render_mapping_chart(vol_level):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    x = list(range(101))
    ax.fill_between(x, x, alpha=0.18, color='#26a69a')
    ax.plot(x, x, color='#26a69a', linewidth=2.5, label='Volume %')
    ax.scatter([vol_level], [vol_level], color='#ef5350', s=110, zorder=5, label='Current Position')
    ax.set_xlabel("Distance (normalized 0–100)", fontsize=10)
    ax.set_ylabel("Volume (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def render_history_chart(history):
    fig, ax = plt.subplots(figsize=(4.4, 1.6))
    labels = [f"t{i}" for i in range(len(history))]
    ax.bar(labels, history, color='#26a69a', width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Vol %", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

VOLUME_SMOOTHING_ALPHA = 0.80  # higher = more responsive
APPLY_VOLUME_MIN_INTERVAL_S = 0.03
APPLY_VOLUME_MIN_DELTA = 1
GESTURE_HOLD_SECONDS = 0.12
PROCESS_SCALE = 0.45  # <1.0 = faster, slightly less detailed

# Beauty filters disabled for lowest latency.
BEAUTY_BLUR_STRENGTH = 0.0
BEAUTY_BRIGHTNESS = 0
BEAUTY_CONTRAST = 1.0
BEAUTY_GAMMA = 1.0
BEAUTY_SATURATION = 1.0


def _open_camera(index: int = 0):
    """Open and warm up the webcam.

    Keeping the handle in st.session_state avoids slow reopen
    on every Streamlit rerun.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            # Lower-latency capture on many webcams (especially Windows).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Warm-up a few frames (some cameras take a moment).
        for _ in range(6):
            cap.read()
            time.sleep(0.02)

        return cap
    except Exception:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None


def _render_inactive_state():
    status_box.markdown("""
    <div class="status-box">
        <b>Camera:</b> Inactive<br>
        <b>Hands:</b> 0<br>
        <b>FPS:</b> 0<br>
        <b>Model:</b> Not Running
    </div>
    """, unsafe_allow_html=True)

    info_box.markdown("""
    <div class="info-grid">
        <div class="info-box">0<br>Landmarks</div>
        <div class="info-box">0<br>Connections</div>
        <div class="info-box">—<br>Resolution</div>
        <div class="info-box">0 ms<br>Latency</div>
    </div>
    """, unsafe_allow_html=True)

    gesture_info.markdown("""
    <div class="status-box">
        <b>Gesture:</b> None<br>
        <b>Distance:</b> 0 px<br>
        <b>Volume:</b> 0%<br>
        <b>Status:</b> Inactive
    </div>
    """, unsafe_allow_html=True)

    info_box.markdown("""
    <div class="info-grid">
        <div class="info-box">0<br>Landmarks</div>
        <div class="info-box">0<br>Connections</div>
        <div class="info-box">0×0<br>Frame Size</div>
        <div class="info-box">0 ms<br>Latency</div>
    </div>
    <div class="status-box" style="margin-top:12px;">
        <b>Hands Detected:</b> 0<br>
        <b>Model Status:</b> Not Running
    </div>
    """, unsafe_allow_html=True)

    camera_placeholder.info("Camera is stopped")

    last_vol = int(st.session_state.current_volume)
    fig_map = render_mapping_chart(last_vol)
    mapping_chart_placeholder.pyplot(fig_map, use_container_width=True)
    plt.close(fig_map)

    current_vol_placeholder.markdown(f"""
    <div class="volume-card">
        <div class="volume-number">{last_vol}</div>
        <div class="volume-unit">%</div>
        <div>
            <span class="badge-active">⏸ Paused</span>
            <span class="badge-synced">⟳ Synced</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    history_chart_placeholder.info("Start the camera to view volume history")



def _camera_step():
    if st.session_state.cap is None:
        st.session_state.cap = _open_camera(0)
        if st.session_state.cap is None:
            st.error("Camera not accessible")
            st.session_state.camera_on = False
            _render_inactive_state()
            return

    with mp_hands.Hands(
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
        max_num_hands=max_hands,
    ) as hands:

        success, frame = st.session_state.cap.read()
        if not success:
            st.error("Camera not accessible")
            st.session_state.camera_on = False
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
            _render_inactive_state()
            return

        frame = cv2.flip(frame, 1)

        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        st.session_state.frame_idx += 1
        hand_count = 0
        gesture_text = "No gesture detected"
        distance_value = 0
        volume_level = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                hand_count += 1

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                h, w, c = frame.shape
                distance, thumb_pos, index_pos = calculate_distance(thumb_tip, index_tip, w, h)
                distance_value = distance

                gesture_name, color, gesture_type = classify_gesture(distance, hand_landmarks, w, h)
                gesture_text = gesture_name

                if gesture_type == "volume":
                    raw_volume = map_distance_to_volume(distance, min_dist=20, max_dist=150)
                else:
                    raw_volume = 0 if gesture_type == "mute" else 100

                st.session_state.smoothed_volume = smooth_volume(
                    st.session_state.smoothed_volume, raw_volume, alpha=0.2
                )
                volume_level = int(round(st.session_state.smoothed_volume))

                set_system_volume(volume_level)

                frame = draw_gesture_info(frame, distance, gesture_name,
                                        volume_level, thumb_pos, index_pos, color)

        curr_time = time.time()
        fps = int(1 / (curr_time - st.session_state.prev_time)) if st.session_state.prev_time != 0 else 0
        st.session_state.prev_time = curr_time

        camera_placeholder.image(
            frame,
            channels="BGR",
            use_container_width=True
        )

        if hand_count > 0:
            info_box.markdown(
                f"""
                <div class="info-grid">
                    <div class="info-box">{21 * hand_count}<br>Landmarks</div>
                    <div class="info-box">{15 * hand_count}<br>Connections</div>
                    <div class="info-box">{frame.shape[1]}×{frame.shape[0]}<br>Frame Size</div>
                    <div class="info-box">{int(1000/fps) if fps>0 else 0} ms<br>Latency</div>
                </div>
                <div class="status-box" style="margin-top:12px;">
                    <b>Hands Detected:</b> {hand_count}<br>
                    <b>Model Status:</b> Running
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            info_box.markdown("""
            <div class="info-grid">
                <div class="info-box">0<br>Landmarks</div>
                <div class="info-box">0<br>Connections</div>
                <div class="info-box">0×0<br>Frame Size</div>
                <div class="info-box">0 ms<br>Latency</div>
            </div>
            <div class="status-box" style="margin-top:12px;">
                <b>Hands Detected:</b> 0<br>
                <b>Model Status:</b> Running
            </div>
            """, unsafe_allow_html=True)

        gesture_info.markdown(f"""
        <div class="status-box">
            <b>Gesture:</b> {gesture_text}<br>
            <b>Distance:</b> {int(distance_value)} px<br>
            <b>Volume:</b> {volume_level}%<br>
            <b>Status:</b> {'Active' if hand_count > 0 else 'Inactive'}
        </div>
        """, unsafe_allow_html=True)

        status_box.markdown(
            f"""
            <div class="status-box">
                <b>Camera:</b> Active<br>
                <b>FPS:</b> {fps}<br>
                <b>Model:</b> MediaPipe Hands
            </div>
            """,
            unsafe_allow_html=True,
        )

        print(f"[Camera: Active | Hands: {hand_count} | FPS: {fps} | Gesture: {gesture_text} | Distance: {int(distance_value)}px | Volume: {volume_level}% | Latency: {int(1000/fps) if fps>0 else 0}ms]")

        st.session_state.volume_history.append(volume_level)
        if len(st.session_state.volume_history) > 20:
            st.session_state.volume_history = st.session_state.volume_history[-20:]
        st.session_state.current_volume = int(volume_level)

        is_active = hand_count > 0
        current_vol_placeholder.markdown(f"""
        <div class="volume-card">
            <div class="volume-number">{volume_level}</div>
            <div class="volume-unit">%</div>
            <div>
                <span class="badge-active">{"✓ Active" if is_active else "⏸ Paused"}</span>
                <span class="badge-synced">⟳ Synced</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_map = render_mapping_chart(volume_level)
        mapping_chart_placeholder.pyplot(fig_map, use_container_width=True)
        plt.close(fig_map)

        history = st.session_state.volume_history
        if history:
            fig_hist = render_history_chart(history)
            history_chart_placeholder.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
        else:
            history_chart_placeholder.info("No volume history yet — start camera and make gestures!")

        time.sleep(0.01)

while True:
    if st.session_state.camera_on:
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(0)

        with mp_hands.Hands(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            max_num_hands=max_hands
        ) as hands:

            success, frame = st.session_state.cap.read()
            if not success:
                st.error("Camera not accessible")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_count = 0
            gesture_text = "No gesture detected"
            distance_value = 0
            volume_level = 0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    hand_count += 1

                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    h, w, c = frame.shape
                    distance, thumb_pos, index_pos = calculate_distance(thumb_tip, index_tip, w, h)
                    distance_value = distance

                    gesture_name, color, gesture_type = classify_gesture(distance, hand_landmarks, w, h)
                    gesture_text = gesture_name

                    if gesture_type == "volume":
                        raw_volume = map_distance_to_volume(distance, min_dist=20, max_dist=150)
                    else:
                        raw_volume = 0 if gesture_type == "mute" else 100

                    st.session_state.smoothed_volume = smooth_volume(
                        st.session_state.smoothed_volume, raw_volume, alpha=0.2
                    )
                    volume_level = int(round(st.session_state.smoothed_volume))

                    set_system_volume(volume_level)

                    frame = draw_gesture_info(frame, distance, gesture_name,
                                            volume_level, thumb_pos, index_pos, color)

            curr_time = time.time()
            fps = int(1 / (curr_time - st.session_state.prev_time)) if st.session_state.prev_time != 0 else 0
            st.session_state.prev_time = curr_time

            camera_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

            if hand_count > 0:
                info_box.markdown(
                    f"""
                    <div class="info-grid">
                        <div class="info-box">{21 * hand_count}<br>Landmarks</div>
                        <div class="info-box">{15 * hand_count}<br>Connections</div>
                        <div class="info-box">{frame.shape[1]}×{frame.shape[0]}<br>Frame Size</div>
                        <div class="info-box">{int(1000/fps) if fps>0 else 0} ms<br>Latency</div>
                    </div>
                    <div class="status-box" style="margin-top:12px;">
                        <b>Hands Detected:</b> {hand_count}<br>
                        <b>Model Status:</b> Running
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                info_box.markdown("""
                <div class="info-grid">
                    <div class="info-box">0<br>Landmarks</div>
                    <div class="info-box">0<br>Connections</div>
                    <div class="info-box">0×0<br>Frame Size</div>
                    <div class="info-box">0 ms<br>Latency</div>
                </div>
                <div class="status-box" style="margin-top:12px;">
                    <b>Hands Detected:</b> 0<br>
                    <b>Model Status:</b> Running
                </div>
                """, unsafe_allow_html=True)

            gesture_info.markdown(f"""
            <div class="status-box">
                <b>Gesture:</b> {gesture_text}<br>
                <b>Distance:</b> {int(distance_value)} px<br>
                <b>Volume:</b> {volume_level}%<br>
                <b>Status:</b> {'Active' if hand_count > 0 else 'Inactive'}
            </div>
            """, unsafe_allow_html=True)

            status_box.markdown(
                f"""
                <div class="status-box">
                    <b>Camera:</b> Active<br>
                    <b>FPS:</b> {fps}<br>
                    <b>Model:</b> MediaPipe Hands
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state.volume_history.append(volume_level)
            if len(st.session_state.volume_history) > 20:
                st.session_state.volume_history = st.session_state.volume_history[-20:]

            st.session_state.current_volume = int(volume_level)

            is_active = hand_count > 0
            current_vol_placeholder.markdown(f"""
            <div class="volume-card">
                <div class="volume-number">{volume_level}</div>
                <div class="volume-unit">%</div>
                <div>
                    <span class="badge-active">{"✓ Active" if is_active else "⏸ Paused"}</span>
                    <span class="badge-synced">⟳ Synced</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig_map = render_mapping_chart(volume_level)
            mapping_chart_placeholder.pyplot(fig_map, use_container_width=True)
            plt.close(fig_map)

            history = st.session_state.volume_history
            if history:
                fig_hist = render_history_chart(history)
                history_chart_placeholder.pyplot(fig_hist, use_container_width=True)
                plt.close(fig_hist)
            else:
                history_chart_placeholder.info("No volume history yet — start camera and make gestures!")

            time.sleep(0.01)

    else:
        if st.session_state.cap:
            st.session_state.cap.release()
            st.session_state.cap = None

        status_box.markdown("""
        <div class="status-box">
            <b>Camera:</b> Inactive<br>
            <b>FPS:</b> 0<br>
            <b>Model:</b> Not Running
        </div>
        """, unsafe_allow_html=True)

        gesture_info.markdown("""
        <div class="status-box">
            <b>Gesture:</b> None<br>
            <b>Distance:</b> 0 px<br>
            <b>Volume:</b> 0%<br>
            <b>Status:</b> Inactive
        </div>
        """, unsafe_allow_html=True)

        camera_placeholder.info("Camera is stopped")

        last_vol = st.session_state.current_volume
        fig_map = render_mapping_chart(last_vol)
        mapping_chart_placeholder.pyplot(fig_map, use_container_width=True)
        plt.close(fig_map)

        current_vol_placeholder.markdown(f"""
        <div class="volume-card">
            <div class="volume-number">{last_vol}</div>
            <div class="volume-unit">%</div>
            <div>
                <span class="badge-active">⏸ Paused</span>
                <span class="badge-synced">⟳ Synced</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        history = st.session_state.volume_history
        if history:
            fig_hist = render_history_chart(history)
            history_chart_placeholder.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)
        else:
            history_chart_placeholder.info("No volume history yet — start camera and make gestures!")

        time.sleep(0.2)

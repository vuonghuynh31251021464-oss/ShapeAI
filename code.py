import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import time

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShapeAI",
    page_icon="🔷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Be+Vietnam+Pro:wght@300;400;500;600&family=Caveat:wght@600&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

/* Background */
.stApp {
    background: #f7f3ee;
    background-image:
        radial-gradient(circle at 20% 20%, rgba(14,90,182,0.06) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(0,200,220,0.05) 0%, transparent 50%);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── LOGO & HEADER ── */
.hero-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2.5rem 0 1rem;
    gap: 0;
}
.logo-svg { width: 90px; height: 90px; }
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #0d1b3e;
    letter-spacing: -1px;
    line-height: 1;
    margin: 0.4rem 0 0;
}
.app-title span {
    color: #1a8cff;
    font-style: italic;
}
.app-tagline {
    font-family: 'Caveat', cursive;
    font-size: 1.25rem;
    color: #5a6a8a;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}

/* ── DIVIDER ── */
.brush-divider {
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #1a8cff 30%, #00c8dc 70%, transparent);
    border-radius: 2px;
    margin: 1.5rem 0;
    opacity: 0.5;
}

/* ── SECTION TITLES ── */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0d1b3e;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-hint {
    font-size: 0.82rem;
    color: #8a94a6;
    font-weight: 300;
    margin-bottom: 0.8rem;
    font-style: italic;
}

/* ── CANVAS WRAPPER ── */
.canvas-frame {
    border: 2px solid #d4dae8;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,27,62,0.08), 0 2px 8px rgba(13,27,62,0.04);
    background: #ffffff;
    position: relative;
}
.canvas-frame::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(26,140,255,0.03), transparent);
    pointer-events: none;
    z-index: 1;
}

/* ── BUTTON ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0d1b3e 0%, #1a4480 60%, #1a8cff 100%);
    color: #ffffff;
    font-family: 'Be Vietnam Pro', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(26,140,255,0.3);
    margin-top: 0.5rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(26,140,255,0.4);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f6ff 100%);
    border: 1.5px solid #c8d8f0;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(13,27,62,0.08);
    margin-top: 1rem;
    animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-shape-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0d1b3e;
    letter-spacing: -0.5px;
}
.result-shape-name span {
    color: #1a8cff;
    font-style: italic;
}
.result-confidence {
    font-family: 'Be Vietnam Pro', sans-serif;
    font-size: 0.9rem;
    color: #5a6a8a;
    font-weight: 400;
    margin-top: 0.25rem;
}
.result-emoji {
    font-size: 2.8rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.confidence-bar-wrap {
    margin-top: 0.8rem;
    background: #e4ecfa;
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #1a8cff, #00c8dc);
    transition: width 0.8s ease;
}

/* ── TIPS CARD ── */
.tips-card {
    background: #fff8f0;
    border: 1.5px solid #f0dcc0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1.2rem;
    font-size: 0.83rem;
    color: #7a5a3a;
    line-height: 1.7;
}
.tips-card b {
    font-family: 'Playfair Display', serif;
    color: #5a3a1a;
    font-size: 0.9rem;
}

/* ── PROGRESS / SPINNER ── */
.stSpinner > div {
    border-color: #1a8cff transparent transparent transparent !important;
}

/* Tool selector */
.stSelectbox label, .stSlider label {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 0.83rem !important;
    color: #5a6a8a !important;
    font-weight: 500 !important;
}

/* Class pills */
.class-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.class-pill {
    background: #e8f0fe;
    color: #1a4480;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 500;
    font-family: 'Be Vietnam Pro', sans-serif;
    border: 1px solid #c8d8f0;
}
.class-pill-active {
    background: linear-gradient(135deg, #1a4480, #1a8cff);
    color: white;
    border-color: transparent;
}

/* all-probs table */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.35rem 0;
    font-size: 0.82rem;
}
.prob-name {
    width: 80px;
    font-family: 'Be Vietnam Pro', sans-serif;
    color: #0d1b3e;
    font-weight: 500;
}
.prob-bar-wrap {
    flex: 1;
    background: #e4ecfa;
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #1a8cff, #00c8dc);
}
.prob-pct {
    width: 38px;
    text-align: right;
    color: #5a6a8a;
    font-size: 0.78rem;
}
</style>
""", unsafe_allow_html=True)

# ─── LOGO SVG ─────────────────────────────────────────────────
LOGO_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" class="logo-svg">
  <defs>
    <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#00c8dc;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0d1b3e;stop-opacity:1" />
    </linearGradient>
  </defs>
  <!-- outer diamond / kite -->
  <polygon points="100,18 172,80 100,172 28,80"
           fill="none" stroke="url(#g1)" stroke-width="8" stroke-linejoin="round"/>
  <!-- inner triangle -->
  <polygon points="100,30 160,90 40,90"
           fill="none" stroke="url(#g1)" stroke-width="7" stroke-linejoin="round"/>
  <!-- circle inside -->
  <circle cx="100" cy="105" r="28"
          fill="none" stroke="url(#g1)" stroke-width="7"/>
  <!-- nodes -->
  <circle cx="100" cy="18"  r="6" fill="#00c8dc"/>
  <circle cx="172" cy="80" r="6" fill="#4a90d9"/>
  <circle cx="100" cy="172" r="6" fill="#0d1b3e"/>
  <circle cx="28"  cy="80" r="6" fill="#1a8cff"/>
</svg>
"""

# ─── HERO HEADER ──────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrapper">
  {LOGO_SVG}
  <div class="app-title">Shape<span>AI</span></div>
  <div class="app-tagline">✦ Vẽ một hình — AI sẽ đoán ngay ✦</div>
</div>
<div class="brush-divider"></div>
""", unsafe_allow_html=True)

# ─── MODEL TRAINING (cached) ──────────────────────────────────
CLASS_NAMES = ['Hình tròn', 'Hình vuông', 'Hình tam giác',
               'Hình chữ nhật', 'Hình elip', 'Ngôi sao']
CLASS_ICONS = ['⭕', '🟦', '🔺', '▬', '🫧', '⭐']
CLASS_EN    = ['circle', 'square', 'triangle', 'rectangle', 'ellipse', 'star']

@st.cache_resource(show_spinner=False)
def load_model():
    """Build & train model on synthetic geometric data."""
    def make_dataset(n=6000, sz=64):
        X, y = [], []
        for _ in range(n):
            img   = np.zeros((sz, sz, 3), dtype=np.uint8)
            label = np.random.randint(0, 6)
            col   = tuple(np.random.randint(60, 240, 3).tolist())
            bg    = tuple(np.random.randint(200, 255, 3).tolist())
            img[:] = bg
            if label == 0:
                r = np.random.randint(14, 26)
                cv2.circle(img, (32, 32), r, col, -1)
            elif label == 1:
                s = np.random.randint(20, 36)
                cv2.rectangle(img,(32-s//2,32-s//2),(32+s//2,32+s//2),col,-1)
            elif label == 2:
                pts = np.array([[32,10],[10,52],[54,52]], np.int32)
                cv2.fillPoly(img, [pts], col)
            elif label == 3:
                cv2.rectangle(img,(10,22),(54,42),col,-1)
            elif label == 4:
                ang = np.random.randint(0, 90)
                cv2.ellipse(img,(32,32),(24,13),ang,0,360,col,-1)
            elif label == 5:
                pts5 = []
                for i in range(5):
                    a = i * 2*np.pi/5 - np.pi/2
                    pts5.append([int(32+22*np.cos(a)), int(32+22*np.sin(a))])
                    a2 = (i+0.5)*2*np.pi/5 - np.pi/2
                    pts5.append([int(32+10*np.cos(a2)), int(32+10*np.sin(a2))])
                cv2.fillPoly(img, [np.array(pts5, np.int32)], col)
            X.append(img); y.append(label)
        X = np.array(X, dtype='float32') / 255.0
        y = to_categorical(np.array(y), 6)
        return X, y

    X, y = make_dataset(6000)
    split = int(len(X)*0.85)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    m = Sequential([
        Conv2D(32,(3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(6, activation='softmax')
    ])
    m.compile('adam','categorical_crossentropy',metrics=['accuracy'])
    m.fit(X_tr, y_tr, epochs=8, batch_size=64,
          validation_data=(X_va, y_va), verbose=0)
    return m

# ─── LOADING SCREEN ───────────────────────────────────────────
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if not st.session_state.model_ready:
    with st.spinner("🎨 Đang khởi động ShapeAI — huấn luyện mô hình lần đầu (~30 giây)…"):
        model = load_model()
    st.session_state.model_ready = True
    st.rerun()
else:
    model = load_model()

# ─── DRAWING TOOLS ────────────────────────────────────────────
st.markdown('<div class="section-label">🖊️ Bảng vẽ của bạn</div>', unsafe_allow_html=True)
st.markdown('<div class="section-hint">Vẽ tự do bất kỳ hình học nào — tròn, vuông, tam giác, chữ nhật, elip hay ngôi sao</div>', unsafe_allow_html=True)

col_tool, col_size, col_color = st.columns([2, 2, 1])
with col_tool:
    draw_mode = st.selectbox("Chế độ vẽ", ["freedraw", "line"], 
                              format_func=lambda x: "✏️ Vẽ tự do" if x=="freedraw" else "📏 Đường thẳng",
                              label_visibility="collapsed")
with col_size:
    stroke_w = st.slider("Độ dày nét", 4, 22, 10, label_visibility="collapsed")
with col_color:
    stroke_c = st.color_picker("Màu", "#1a4480", label_visibility="collapsed")

# Canvas
st.markdown('<div class="canvas-frame">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_w,
    stroke_color=stroke_c,
    background_color="#ffffff",
    height=320,
    width=680,
    drawing_mode=draw_mode,
    key="canvas",
    display_toolbar=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# Supported shapes pills
st.markdown("""
<div class="class-pills">
  <span class="class-pill">⭕ Hình tròn</span>
  <span class="class-pill">🟦 Hình vuông</span>
  <span class="class-pill">🔺 Tam giác</span>
  <span class="class-pill">▬ Chữ nhật</span>
  <span class="class-pill">🫧 Elip</span>
  <span class="class-pill">⭐ Ngôi sao</span>
</div>
""", unsafe_allow_html=True)

# ─── PREDICT BUTTON ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
btn_predict = st.button("🔍 Đoán hình ngay!", use_container_width=True)

# ─── PREDICTION LOGIC ─────────────────────────────────────────
if btn_predict:
    if canvas_result.image_data is None:
        st.warning("✏️ Hãy vẽ một hình trước nhé!")
    else:
        img_arr = canvas_result.image_data.astype(np.uint8)
        # Check if canvas is (nearly) blank
        alpha = img_arr[:, :, 3] if img_arr.shape[2] == 4 else None
        if alpha is not None and alpha.sum() < 500:
            st.warning("✏️ Bảng vẽ trống! Hãy vẽ một hình trước.")
        else:
            with st.spinner("🤖 AI đang phân tích…"):
                # Convert RGBA → RGB white bg
                rgba = Image.fromarray(img_arr, 'RGBA')
                bg   = Image.new('RGB', rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[3])
                rgb  = bg.resize((64, 64), Image.LANCZOS)
                inp  = np.array(rgb, dtype='float32') / 255.0
                inp  = inp.reshape(1, 64, 64, 3)
                preds = model.predict(inp, verbose=0)[0]
                idx   = int(np.argmax(preds))
                conf  = float(preds[idx]) * 100
                time.sleep(0.3)  # small dramatic pause

            # ── RESULT CARD ──
            icon = CLASS_ICONS[idx]
            name = CLASS_NAMES[idx]
            bar_w = int(conf)

            st.markdown(f"""
<div class="result-card">
  <div class="result-emoji">{icon}</div>
  <div class="result-shape-name">Đây là <span>{name}</span>!</div>
  <div class="result-confidence">Độ tự tin: {conf:.1f}%</div>
  <div class="confidence-bar-wrap">
    <div class="confidence-bar-fill" style="width:{bar_w}%"></div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── ALL PROBABILITIES ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label" style="font-size:0.95rem;">📊 Phân tích chi tiết</div>', unsafe_allow_html=True)
            
            prob_rows = ""
            sorted_idx = np.argsort(preds)[::-1]
            for i in sorted_idx:
                p = float(preds[i]) * 100
                bw = int(p)
                bold = "font-weight:700;color:#0d1b3e;" if i == idx else ""
                prob_rows += f"""
<div class="prob-row">
  <div class="prob-name" style="{bold}">{CLASS_ICONS[i]} {CLASS_NAMES[i].replace('Hình ','')}</div>
  <div class="prob-bar-wrap">
    <div class="prob-bar-fill" style="width:{bw}%;{'opacity:1' if i==idx else 'opacity:0.45'}"></div>
  </div>
  <div class="prob-pct" style="{bold}">{p:.1f}%</div>
</div>"""
            
            st.markdown(f'<div style="background:#fff;border:1.5px solid #d4dae8;border-radius:12px;padding:1rem 1.25rem;">{prob_rows}</div>', unsafe_allow_html=True)

# ─── TIPS ─────────────────────────────────────────────────────
st.markdown("""
<div class="tips-card">
  <b>💡 Mẹo vẽ tốt hơn</b><br>
  • Vẽ hình <b>lớn</b>, chiếm nhiều diện tích bảng vẽ<br>
  • Hình tròn & elip: vẽ thành vòng khép kín<br>
  • Tam giác: vẽ 3 cạnh rõ ràng<br>
  • Ngôi sao: vẽ kiểu David star hoặc 5 cánh<br>
  • Dùng nét bút dày hơn để AI nhận dạng chính xác hơn
</div>
""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2.5rem;padding:1rem;
            font-family:'Be Vietnam Pro',sans-serif;font-size:0.78rem;
            color:#9aa3b4;border-top:1px solid #e0e6f0;">
  <span style="font-family:'Playfair Display',serif;font-weight:700;color:#0d1b3e;">Shape<span style="color:#1a8cff;font-style:italic">AI</span></span>
  &nbsp;·&nbsp; CNN · TensorFlow · Streamlit
  &nbsp;·&nbsp; Made with ♥ in Việt Nam
</div>
""", unsafe_allow_html=True)


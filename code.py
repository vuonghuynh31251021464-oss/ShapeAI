import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="ShapeAI",
    page_icon="🔷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS (giữ nguyên của bạn) ─────────────────────────────────
st.markdown("""[Your long CSS here - giữ nguyên như cũ]""", unsafe_allow_html=True)

# ─── LOGO & HEADER (giữ nguyên) ─────────────────────────────────
# ... (phần LOGO_SVG và hero header của bạn giữ nguyên)

# ─── CLASS NAMES ──────────────────────────────────────────────
CLASS_NAMES = ['Hình tròn', 'Hình vuông', 'Hình tam giác', 'Hình chữ nhật', 'Hình elip', 'Ngôi sao']
CLASS_ICONS = ['⭕', '🟦', '🔺', '▬', '🫧', '⭐']

# ─── FEATURE EXTRACTION (CẢI TIẾN) ─────────────────────────────
def extract_features(img_gray):
    sz = 64
    img = cv2.resize(img_gray, (sz, sz))
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    features = []
    small = cv2.resize(bw, (16, 16)).flatten().astype(float) / 255.0
    features.extend(small)
    
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / (h + 1e-5)
        extent = float(area) / (w * h + 1e-5)
        
        hull = cv2.convexHull(cnt)
        solidity = float(area) / (cv2.contourArea(hull) + 1e-5)
        
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        n_vertices = len(approx)
        
        features.extend([circularity, aspect_ratio, extent, solidity, n_vertices/20.0, area/(sz*sz)])
        
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-8)
        features.extend(hu_log.tolist())
    else:
        features.extend([0.0] * 13)
    
    # Quadrants
    h, w = bw.shape
    for i in range(2):
        for j in range(2):
            quad = bw[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
            features.append(quad.sum() / (quad.size * 255.0 + 1e-5))
    
    return np.array(features, dtype=float)

# ─── MAKE DATASET ─────────────────────────────────────────────
def make_dataset(n=4000):
    X, y = [], []
    sz = 64
    for _ in range(n):
        img = np.zeros((sz, sz, 3), dtype=np.uint8)
        img[:] = 255
        label = np.random.randint(0, 6)
        col = tuple(np.random.randint(50, 220, 3).tolist())
        cx, cy = sz//2 + np.random.randint(-6,7), sz//2 + np.random.randint(-6,7)
        
        if label == 0:    # Circle
            cv2.circle(img, (cx, cy), np.random.randint(16,27), col, -1)
        elif label == 1:  # Square
            s = np.random.randint(22,36)
            cv2.rectangle(img, (cx-s//2, cy-s//2), (cx+s//2, cy+s//2), col, -1)
        elif label == 2:  # Triangle
            h_ = np.random.randint(26,40)
            pts = np.array([[cx, cy-h_//2], [cx-h_//2, cy+h_//2], [cx+h_//2, cy+h_//2]], np.int32)
            cv2.fillPoly(img, [pts], col)
        elif label == 3:  # Rectangle
            cv2.rectangle(img, (cx-22, cy-12), (cx+22, cy+12), col, -1)
        elif label == 4:  # Ellipse
            cv2.ellipse(img, (cx, cy), (24,14), np.random.randint(0,80), 0, 360, col, -1)
        elif label == 5:  # Star
            pts = []
            for i in range(5):
                a = i*2*np.pi/5 - np.pi/2
                pts.append([int(cx + 23*np.cos(a)), int(cy + 23*np.sin(a))])
                a2 = (i+0.5)*2*np.pi/5 - np.pi/2
                pts.append([int(cx + 10*np.cos(a2)), int(cy + 10*np.sin(a2))])
            cv2.fillPoly(img, [np.array(pts, np.int32)], col)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(extract_features(gray))
        y.append(label)
    return np.array(X), np.array(y)

# ─── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists('shapeai_model.pkl') and os.path.exists('scaler.pkl'):
        return joblib.load('shapeai_model.pkl'), joblib.load('scaler.pkl')
    
    with st.spinner("🎨 Đang huấn luyện ShapeAI (khoảng 8-12 giây)..."):
        X, y = make_dataset(4000)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        clf.fit(X_scaled, y)
        
        joblib.dump(clf, 'shapeai_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
    return clf, scaler

# ─── MAIN APP ─────────────────────────────────────────────────
model, scaler = load_model()

# Phần vẽ canvas và logic dự đoán của bạn (giữ nguyên)...

st.success("✅ ShapeAI đã sẵn sàng!")

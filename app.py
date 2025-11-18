"""
YOLOv7-tiny Custom Detector
Deployed on Streamlit Cloud
Detects: cheerios, soup, candle
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os

# ========================= CONFIG =========================
MODEL_PATH = "yolov7-tiny.pt"          # Your uploaded model file
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASSES = ['cheerios', 'soup', 'candle']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR

# ========================= LOAD MODEL (SAFE WAY) =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found! Make sure '{MODEL_PATH}' is in the repo root.")
        st.stop()

    # Load the full checkpoint
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model = ckpt['model']
    model = model.float().eval()
    
    # Remove DataParallel wrapper if present
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    st.success("Model loaded successfully!")
    return model

# ========================= PREPROCESS =========================
def preprocess(img_pil):
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Letterbox resize
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    img_input = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    dw = (IMG_SIZE - new_w) // 2
    dh = (IMG_SIZE - new_h) // 2
    img_input[dh:dh+new_h, dw:dw+new_w] = img_resized
    
    # To tensor
    img_input = img_input.transpose(2, 0, 1)  # HWC → CHW
    img_input = torch.from_numpy(img_input).unsqueeze(0).float() / 255.0
    return img_input, (dw, dh, r), (h, w)

# ========================= POSTPROCESS =========================
def postprocess(pred, dw, dh, r, orig_shape):
    pred = pred[0].cpu().numpy()
    boxes, scores, class_ids = [], [], []
    h, w = orig_shape
    
    for *box, conf, cls in pred:
        if conf < CONF_THRESHOLD:
            continue
        x1 = int((box[0] - dw) / r)
        y1 = int((box[1] - dh) / r)
        x2 = int((box[2] - dw) / r)
        y2 = int((box[3] - dh) / r)
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(int(cls))
    return boxes, scores, class_ids

# ========================= DRAW =========================
def draw_boxes(img, boxes, scores, class_ids):
    img = img.copy()
    for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, class_ids):
        color = COLORS[cls_id]
        label = f"{CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# ========================= MAIN APP =========================
st.set_page_config(page_title="YOLOv7 Grocery Detector", layout="centered")
st.title("YOLOv7-tiny Object Detector")
st.markdown("**Detects:** `cheerios` • `soup` • `candle`")

# Load model
with st.spinner("Loading YOLOv7 model..."):
    model = load_model()

# Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting objects..."):
        img_input, pad_info, orig_shape = preprocess(image)
        with torch.no_grad():
            pred = model(img_input)[0]

        boxes, scores, class_ids = postprocess(pred, *pad_info, orig_shape)
        
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        result_img = draw_boxes(img_bgr, boxes, scores, class_ids)
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

        st.image(result_pil, caption="Detection Result", use_column_width=True)
        
        if len(boxes) == 0:
            st.info("No objects detected.")
        else:
            st.success(f"Found {len(boxes)} object(s)!")
else:
    st.info("Upload an image to start detecting!")
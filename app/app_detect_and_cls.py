#!/usr/bin/env python3
# screen_or_video_detect_with_cls_arrow.py
# pip install ultralytics timm opencv-python torchvision torch pillow mss

import time
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import torch
import timm
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms

# ---- GUI (tkinter) ----
import tkinter as tk
from tkinter import filedialog

import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    # đang chạy trong .exe
    BASE_DIR = Path(sys._MEIPASS)
else:
    # đang chạy file .py
    BASE_DIR = Path(__file__).resolve().parent
# ====== CẤU HÌNH NGƯỜI DÙNG ======
WEIGHTS   = str(BASE_DIR / "models" / "best_yolo.pt")
CONF = 0.45
IOU = 0.45
IMGSZ = 736
CLASSES = None

# Classifier config
CLS_CKPT  = str(BASE_DIR / "models" / "best_cls.pt")
CLS_ARCH = "convnext_tiny"
CLS_SIZE = 224
USE_LETTERBOX_REFLECT = True

YOLO_DEVICE = 0 if torch.cuda.is_available() else 'cpu'
CLS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Screen capture
MONITOR_INDEX = 1
CROP = None

# Visual
THICKNESS = 2
FONT_SCALE = 0.6
CLS_LABEL_COLOR = (0, 0, 255)  # red
CROP_TOP_PCT = 0.05   # 5% from top of bbox crop
CROP_BOTTOM_PCT = 0.20 # 20% from bottom of bbox crop

# Safety
MIN_CROP_SIDE = 10
BATCH_SIZE = 32


# ---------- Letterbox (reflect) ----------
class LetterboxSquareTrainStyle:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img
        s = min(self.size / w, self.size / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        pad_l = (self.size - nw) // 2
        pad_t = (self.size - nh) // 2
        pad_r = self.size - nw - pad_l
        pad_b = self.size - nh - pad_t
        return F.pad(img, (pad_l, pad_t, pad_r, pad_b), padding_mode="reflect")


def letterbox_reflect_cv(im: np.ndarray, new_shape: int):
    if im is None or im.size == 0:
        return im
    h0, w0 = im.shape[:2]
    new_unpad = (new_shape, new_shape)
    s = min(new_unpad[0] / w0, new_unpad[1] / h0)
    new_w, new_h = int(round(w0 * s)), int(round(h0 * s))
    if (new_w, new_h) != (w0, h0):
        im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        im_resized = im.copy()
    pad_l = (new_shape - new_w) // 2
    pad_t = (new_shape - new_h) // 2
    pad_r = new_shape - new_w - pad_l
    pad_b = new_shape - new_h - pad_t
    im_padded = cv2.copyMakeBorder(im_resized, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REFLECT_101)
    return im_padded


# ---------- Classifier loading ----------
def strip_module_prefix(sd: dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_classifier(ckpt_path: str, arch: str, img_size: int):
    ck = safe_torch_load(ckpt_path, map_location="cpu")
    classes = ck.get("classes", [str(i) for i in range(12)])
    n = len(classes)
    model = timm.create_model(arch, pretrained=False, num_classes=n)
    sd = strip_module_prefix(ck["model"])
    model.load_state_dict(sd, strict=True)
    model.to(CLS_DEVICE).eval()
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return model, classes, tf


# ---------- GUI chọn nguồn ----------
def choose_source():
    """
    Mở một cửa sổ nhỏ:
      - Nút 'Quay màn hình'
      - Nút 'Chọn video'
    Trả về (mode, video_path or None)
    """
    root = tk.Tk()
    root.title("Chọn nguồn")
    root.geometry("320x140")
    root.resizable(False, False)

    choice = {"mode": None, "path": None}

    def select_screen():
        choice["mode"] = "screen"
        root.destroy()

    def select_video():
        path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mkv;*.mov;*.m4v"),
                ("All files", "*.*")
            ]
        )
        if path:
            choice["mode"] = "video"
            choice["path"] = path
            root.destroy()

    label = tk.Label(root, text="Chọn nguồn input:", font=("Segoe UI", 11))
    label.pack(pady=10)

    btn1 = tk.Button(root, text="Quay theo màn hình", command=select_screen, width=20)
    btn1.pack(pady=2)

    btn2 = tk.Button(root, text="Chọn file video", command=select_video, width=20)
    btn2.pack(pady=2)

    info = tk.Label(root, text="Nhấn ESC để đóng cửa sổ.", font=("Segoe UI", 8))
    info.pack(pady=5)

    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()

    return choice["mode"], choice["path"]


# ---------- Vẽ mũi tên CLS (trên bbox, mờ, cố định kích thước) ----------
def draw_cls_arrow(
    annotated,
    box,
    label,
    color=(0, 0, 255),
    alpha=0.45,
    gap=18,          # khoảng giữa box và mũi tên
    arrow_w=100,      # RỘNG cố định của mũi tên
    arrow_h=14       # CAO cố định của mũi tên
):
    """
    Mũi tên CỐ ĐỊNH kích thước, luôn nằm TRÊN bbox, chĩa xuống.
    Không viền, mờ mờ, text nằm trên tam giác.
    """
    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2

    # ---- vị trí Y cố định ----
    tip_y = max(0, y1 - gap)         # đỉnh dưới mũi tên
    base_y = max(0, tip_y - arrow_h) # cạnh trên tam giác

    # ---- tọa độ 3 điểm tam giác ----
    half = arrow_w // 2
    left_x  = cx - half
    right_x = cx + half

    # clamp để không vượt khung hình
    H, W = annotated.shape[:2]
    left_x  = max(0, min(left_x,  W-1))
    right_x = max(0, min(right_x, W-1))

    pts = np.array([
        [left_x,  base_y],
        [right_x, base_y],
        [cx,      tip_y]
    ], dtype=np.int32)

    # ---- fill convex polygon (không viền) ----
    overlay = annotated.copy()
    cv2.fillConvexPoly(overlay, pts, color)

    # ---- blend mờ ----
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, dst=annotated)

    # ---- text nằm trên tam giác ----
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    text_x = int(cx - tw / 2)
    text_y = max(5, base_y - 6)

    # tránh tràn biên
    text_x = max(5, min(text_x, W - tw - 5))

    cv2.putText(
        annotated,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )




# ---------- Core inference chung (1 frame) ----------
def run_inference_on_frame(frame, yolo, cls_model, class_names, to_tensor, lp_pil):
    """
    Nhận 1 frame BGR, trả về frame đã annotate.
    Logic detect + crop + cls giữ nguyên, chỉ thay phần vẽ text bằng vẽ mũi tên.
    """
    res = yolo.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        classes=CLASSES,
        device=YOLO_DEVICE,
        verbose=False
    )[0]

    crops = []
    boxes = []
    scores = []

    if res.boxes is not None and res.boxes.shape[0] > 0:
        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else 0.0
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < MIN_CROP_SIDE:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            ch = crop.shape[0]
            top_px = int(round(ch * CROP_TOP_PCT))
            bottom_px = int(round(ch * CROP_BOTTOM_PCT))
            y_start = min(max(0, top_px), ch)
            y_end = min(max(y_start, ch - bottom_px), ch)
            if y_start >= y_end:
                continue
            percent_crop = crop[y_start:y_end, :, :]

            if percent_crop.shape[0] < 8 or percent_crop.shape[1] < 8:
                continue

            crop_rgb = cv2.cvtColor(percent_crop, cv2.COLOR_BGR2RGB)

            try:
                if USE_LETTERBOX_REFLECT and lp_pil is not None:
                    pil = Image.fromarray(crop_rgb)
                    pil_out = lp_pil(pil)
                    in_img = np.array(pil_out)
                else:
                    in_img = letterbox_reflect_cv(crop_rgb, CLS_SIZE)
            except Exception:
                in_img = cv2.resize(crop_rgb, (CLS_SIZE, CLS_SIZE), interpolation=cv2.INTER_LINEAR)

            in_img = np.asarray(in_img, dtype=np.uint8)
            if in_img.ndim != 3 or in_img.shape[0] != CLS_SIZE or in_img.shape[1] != CLS_SIZE:
                in_img = cv2.resize(in_img, (CLS_SIZE, CLS_SIZE), interpolation=cv2.INTER_LINEAR)

            crops.append(in_img)
            boxes.append((x1, y1, x2, y2))
            scores.append(conf)

    preds = [None] * len(crops)
    confs = [None] * len(crops)
    if crops:
        tens = []
        valid_indices = []
        for i, c in enumerate(crops):
            c = np.clip(c, 0, 255).astype(np.uint8)
            try:
                t = to_tensor(c)
            except Exception:
                try:
                    t = to_tensor(Image.fromarray(c))
                except Exception:
                    continue
            if not torch.isfinite(t).all():
                continue
            tens.append(t)
            valid_indices.append(i)

        if tens:
            batch = torch.stack(tens, dim=0).to(CLS_DEVICE)
            with torch.no_grad():
                logits = cls_model(batch)
                probs = torch.softmax(logits, dim=1)
                conf_vals, pred_vals = probs.max(dim=1)
                conf_list = conf_vals.cpu().numpy().tolist()
                pred_list = pred_vals.cpu().numpy().tolist()
            for k, vi in enumerate(valid_indices):
                preds[vi] = int(pred_list[k])
                confs[vi] = float(conf_list[k])

    annotated = res.plot(line_width=THICKNESS, font_size=FONT_SCALE)

    # Vẽ mũi tên CLS ở trên bbox
    for box, pred, conf_val in zip(boxes, preds, confs):
        if pred is None or conf_val is None or not np.isfinite(conf_val):
            continue
        try:
            name = class_names[int(pred)]
        except Exception:
            name = str(int(pred))
        label = f"{name} {conf_val:.2f}"
        draw_cls_arrow(annotated, box, label, CLS_LABEL_COLOR, alpha=0.45)

    return annotated


# ---------- Main: hai mode screen / video ----------
def main():
    yolo = YOLO(WEIGHTS)
    yolo.to(YOLO_DEVICE)
    print(f"[INFO] YOLO loaded on device={YOLO_DEVICE}")

    cls_model, class_names, to_tensor = load_classifier(CLS_CKPT, CLS_ARCH, CLS_SIZE)
    print(f"[INFO] CLS loaded on {CLS_DEVICE}, classes={len(class_names)}")

    lp_pil = LetterboxSquareTrainStyle(CLS_SIZE) if USE_LETTERBOX_REFLECT else None

    mode, video_path = choose_source()
    if mode is None:
        print("[INFO] Không chọn nguồn, thoát.")
        return

    prev_t = time.time()
    fps = 0.0

    if mode == "screen":
        print("[INFO] Mode: SCREEN CAPTURE")
        sct = mss()
        monitors = sct.monitors
        if MONITOR_INDEX >= len(monitors):
            raise ValueError(f"No monitor index {MONITOR_INDEX}")
        mon = monitors[MONITOR_INDEX]
        if CROP is None:
            region = {
                'left': mon['left'],
                'top': mon['top'],
                'width': mon['width'],
                'height': mon['height']
            }
        else:
            region = {
                'left': mon['left'] + CROP['left'],
                'top': mon['top'] + CROP['top'],
                'width': CROP['width'],
                'height': CROP['height']
            }

        try:
            while True:
                img = np.array(sct.grab(region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                annotated = run_inference_on_frame(frame, yolo, cls_model, class_names, to_tensor, lp_pil)

                now = time.time()
                dt = now - prev_t
                prev_t = now
                fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("YOLO Screen/Video + CLS (arrow)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    break
        finally:
            cv2.destroyAllWindows()
            sct.close()

    elif mode == "video":
        print(f"[INFO] Mode: VIDEO FILE -> {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[ERROR] Không mở được video.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = run_inference_on_frame(frame, yolo, cls_model, class_names, to_tensor, lp_pil)

                now = time.time()
                dt = now - prev_t
                prev_t = now
                fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("YOLO Screen/Video + CLS (arrow)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

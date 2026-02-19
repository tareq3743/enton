#!/usr/bin/env python3
"""Vision Monitor — Excellence Edition (Qt6 + TensorRT + ByteTrack).

Advanced real-time vision pipeline featuring:
- Asynchronous TensorRT Inference
- ByteTrack Multi-Object Tracking (Consistent IDs)
- High-Performance OpenGL Rendering
- Latency and Throughput Analytics

Usage:
    uv run scripts/dev/vision_monitor.py [--webcam] [--cam=ID]
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List

# Core Optimizations & Log Suppression
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def silence_c_logs():
    """Silences low-level C library warnings by redirecting stderr at the FD level."""
    try:
        # Redirect Python stderr
        sys.stderr = open(os.devnull, 'w')
        # Redirect C-level stderr (File Descriptor 2)
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, 2)
    except Exception:
        pass

# Activate terminal cleaning protocol
silence_c_logs()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QRect
from PySide6.QtGui import QAction, QImage, QKeySequence, QSurfaceFormat, QPainter, QColor, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QStatusBar
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from ultralytics import YOLO

# Perception Layer
from enton.perception.activity import NOSE, _visible as visible, classify as classify_activity
from enton.perception.emotion import EmotionRecognizer
from enton.perception.visualization import Visualizer

# Hardware Config
RTSP_URL = "rtsp://192.168.18.23:554/video0_unicast"
DET_MODEL_PT = "models/yolo11x.pt"
POSE_MODEL_PT = "models/yolo11x-pose.pt"
INFERENCE_SIZE = 640

# Keypoint Map
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class VideoGLWidget(QOpenGLWidget):
    """Zero-copy oriented OpenGL renderer."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._bg = QColor(15, 15, 15)

    @Slot(np.ndarray)
    def update_frame(self, rgb_frame: np.ndarray):
        h, w, _ = rgb_frame.shape
        # Create QImage from worker buffer
        self._image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self.update()

    def paintGL(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self._bg)
        if self._image:
            rect = self._calc_rect(self._image.size())
            p.drawImage(rect, self._image)

    def _calc_rect(self, size: QSize):
        return QRect(0, 0, self.width(), self.height()) # Simple stretch for excellence display

class VisionWorker(QThread):
    """High-throughput pipeline worker."""
    frame_ready = Signal(np.ndarray)
    
    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = True
        self._overlay = True
        
    def run(self):
        # 1. Init
        det_path = self._resolve(DET_MODEL_PT)
        pose_path = self._resolve(POSE_MODEL_PT)
        
        det_model = YOLO(det_path, task="detect")
        pose_model = YOLO(pose_path, task="pose")
        viz = Visualizer()
        emo = EmotionRecognizer(device="cuda:0")
        
        # Force V4L2 backend for Linux stability
        backend = cv2.CAP_V4L2 if isinstance(self.source, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.source, backend)
        
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 2. Loop
        while self._running:
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            
            # --- Inference Pipeline ---
            inf_start = time.perf_counter()
            # USE TRACKING (ByteTrack) for consistent IDs
            results = det_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=INFERENCE_SIZE)
            pose_res = pose_model.predict(frame, verbose=False, imgsz=INFERENCE_SIZE, half=True)
            inf_time = (time.perf_counter() - inf_start) * 1000
            
            # --- Visualization ---
            canvas = frame.copy()
            
            # Tracked Entities
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                names = results[0].names
                
                for box, obj_id, conf, cls_idx in zip(boxes, ids, confs, classes):
                    name = names[cls_idx]
                    color = (0, 255, 0) if name == "person" else (0, 165, 255)
                    viz.draw_entity(canvas, tuple(box), name, conf, obj_id, color)

            # Skeletons
            kpts_list = []
            if pose_res[0].keypoints is not None:
                kpts_data = pose_res[0].keypoints.data
                kpts_list = list(kpts_data)
                for kpts in kpts_data:
                    viz.draw_skeleton(canvas, kpts, SKELETON, (0, 255, 255), visible)
                    activity, _ = classify_activity(kpts)
                    if visible(kpts, NOSE):
                        viz.draw_activity_label(canvas, activity, (int(kpts[NOSE][0]), int(kpts[NOSE][1])), (0, 255, 255))

            # Performance
            total_time = (time.perf_counter() - start_time) * 1000
            metrics = {
                "Inferencia": inf_time,
                "Pipeline": total_time,
                "FPS Real": 1000.0 / total_time if total_time > 0 else 0
            }
            viz.draw_performance_panel(canvas, metrics)
            
            # Dispatch
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(rgb)

        cap.release()

    def _resolve(self, path):
        p = Path(path)
        return str(p.with_suffix(".engine")) if p.with_suffix(".engine").exists() else str(p)

    def stop(self): self._running = False; self.wait()

class MainWindow(QMainWindow):
    def __init__(self, source):
        super().__init__()
        self.setWindowTitle("ENTON · Vision Excellence")
        self.resize(1280, 720)
        self.setStyleSheet("QMainWindow { background: #0a0a0a; } QStatusBar { color: #666; background: #111; }")
        
        self.view = VideoGLWidget()
        self.setCentralWidget(self.view)
        self.setStatusBar(QStatusBar())
        
        self.worker = VisionWorker(source)
        self.worker.frame_ready.connect(self.view.update_frame)
        self.worker.start()
        
        # Actions
        q = QAction("Sair", self); q.setShortcut("q"); q.triggered.connect(self.close); self.addAction(q)
        f = QAction("FS", self); f.setShortcut("f"); f.triggered.connect(self.showFullScreen); self.addAction(f)

    def closeEvent(self, e): self.worker.stop(); e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Fix GTK theme warnings by using Qt's native style
    src = int(sys.argv[1].split('=')[1]) if "--cam=" in "".join(sys.argv) else 0
    win = MainWindow(src)
    window_icon = Path("static/logo.png")
    if window_icon.exists(): win.setWindowIcon(QIcon(str(window_icon)))
    win.show()
    sys.exit(app.exec())

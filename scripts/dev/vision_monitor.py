#!/usr/bin/env python3
"""Vision Monitor — Grid Excellence Edition (Qt6 + Multi-Camera + Control).

High-performance real-time vision grid featuring:
- Parallel Multi-Camera Processing (N-Sources)
- Independent TensorRT Inference per Stream
- ByteTrack Persistence across all views
- Dynamic Grid Layout (Auto-Reflow on Connect/Disconnect)
- Remote Camera Control (Zoom/Flash) integration

Usage:
    uv run scripts/dev/vision_monitor.py [--source=URL] [--webcam]
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict
import urllib.request
import threading

# Core Optimizations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def silence_c_logs():
    """Silences low-level C library warnings."""
    try:
        sys.stderr = open(os.devnull, 'w')
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, 2)
    except Exception:
        pass

silence_c_logs()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QRect
from PySide6.QtGui import QAction, QImage, QKeySequence, QSurfaceFormat, QPainter, QColor, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QStatusBar, QGridLayout, QLabel
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from ultralytics import YOLO

# Perception Layer
from enton.perception.activity import NOSE, _visible as visible, classify as classify_activity
from enton.perception.emotion import EmotionRecognizer
from enton.perception.visualization import Visualizer
from enton.core.config import settings

# Keypoint Map
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

class IPWebcamController:
    """Remote control for Android IP Webcam app."""
    def __init__(self, url):
        self.base_url = str(url).rsplit('/', 1)[0]
        self.zoom_level = 0
        self.is_ip_webcam = "10." in str(url) or "192.168" in str(url)

    def _send(self, endpoint):
        if not self.is_ip_webcam: return
        def task():
            try: urllib.request.urlopen(f"{self.base_url}/{endpoint}", timeout=1)
            except: pass
        threading.Thread(target=task, daemon=True).start()

    def set_torch(self, state: bool):
        self._send("enabletorch" if state else "disabletorch")

    def focus(self): self._send("focus")
    
    def zoom(self, delta):
        self.zoom_level = max(0, min(100, self.zoom_level + delta))
        self._send(f"ptz?zoom={self.zoom_level}")

class VideoGLWidget(QOpenGLWidget):
    """High-performance OpenGL renderer for a single stream."""
    def __init__(self, label="Stream", parent=None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._bg = QColor(10, 10, 10)
        self._label = label
        self._is_active = False # For border highlight

    def set_active(self, active: bool):
        self._is_active = active
        self.update()

    @Slot(np.ndarray)
    def update_frame(self, rgb_frame: np.ndarray):
        h, w, _ = rgb_frame.shape
        self._image = QImage(rgb_frame.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self.update()

    def paintGL(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self._bg)
        
        if self._image:
            rect = self._calc_aspect_rect(self._image.size())
            p.drawImage(rect, self._image)
            
            # Active Border
            if self._is_active:
                p.setPen(QColor(0, 255, 0))
                p.drawRect(self.rect().adjusted(1,1,-1,-1))
            
            # Label
            p.setPen(QColor(0, 255, 0))
            p.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
            p.drawText(10, 25, self._label.upper())

    def _calc_aspect_rect(self, size: QSize):
        w_widget, h_widget = self.width(), self.height()
        w_img, h_img = size.width(), size.height()
        aspect_img = w_img / h_img
        
        if w_widget / h_widget > aspect_img:
            h_target = h_widget
            w_target = int(h_widget * aspect_img)
        else:
            w_target = w_widget
            h_target = int(w_widget / aspect_img)
            
        return QRect((w_widget - w_target) // 2, (h_widget - h_target) // 2, w_target, h_target)

class VisionWorker(QThread):
    frame_ready = Signal(np.ndarray)
    status_changed = Signal(bool, str) # status, worker_name
    
    def __init__(self, name, source):
        super().__init__()
        self.name = name
        self.source = source
        self._running = True
        self._overlay = True
        self._is_online = False
        
    def _resolve(self, path):
        p = Path(path)
        return str(p.with_suffix(".engine")) if p.with_suffix(".engine").exists() else str(path)

    def run(self):
        det_model = YOLO(self._resolve("models/yolo11x.pt"), task="detect")
        pose_model = YOLO(self._resolve("models/yolo11x-pose.pt"), task="pose")
        viz = Visualizer()
        
        backend = cv2.CAP_V4L2 if isinstance(self.source, int) else cv2.CAP_FFMPEG
        cap = cv2.VideoCapture(self.source, backend)
        
        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self._running:
            t_start = time.perf_counter()
            ret, frame = cap.read()
            
            # Dynamic Connection Status Logic
            if ret and not self._is_online:
                self._is_online = True
                self.status_changed.emit(True, self.name)
            elif not ret and self._is_online:
                self._is_online = False
                self.status_changed.emit(False, self.name)

            if not ret:
                if not isinstance(self.source, int): 
                    time.sleep(1); cap.open(self.source)
                continue
            
            if self._overlay:
                i_start = time.perf_counter()
                results = det_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=640)
                pose_res = pose_model.predict(frame, verbose=False, imgsz=640, half=True)
                inf_time = (time.perf_counter() - i_start) * 1000
                
                canvas = frame.copy()
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confs = results[0].boxes.conf.cpu().numpy()
                    names = results[0].names
                    for b, tid, c, cl in zip(boxes, ids, confs, results[0].boxes.cls.cpu().numpy().astype(int)):
                        viz.draw_entity(canvas, tuple(b), names[cl], c, tid, (0, 255, 0))

                if pose_res[0].keypoints is not None:
                    for kpts in pose_res[0].keypoints.data:
                        viz.draw_skeleton(canvas, kpts, SKELETON, (0, 255, 255), visible)
                        act, _ = classify_activity(kpts)
                        if visible(kpts, NOSE):
                            viz.draw_activity_label(canvas, act, (int(kpts[NOSE][0]), int(kpts[NOSE][1])), (0, 255, 255))

                metrics = {"Latencia": inf_time, "FPS": 1.0 / (time.perf_counter() - t_start)}
                viz.draw_performance_panel(canvas, metrics, system_status=self.name.upper())
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            self.frame_ready.emit(rgb)
        cap.release()

    def stop(self): self._running = False; self.wait()
    def toggle_overlay(self): self._overlay = not self._overlay

class MainWindow(QMainWindow):
    def __init__(self, cam_sources: Dict[str, str | int]):
        super().__init__()
        self.setWindowTitle("ENTON · Vision Command (Dynamic Grid)")
        self.resize(1280, 720) # Standard HD window size
        self.setStyleSheet("QMainWindow { background: #050505; } QStatusBar { color: #666; background: #111; }")
        
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.grid = QGridLayout(self.central)
        self.grid.setContentsMargins(2, 2, 2, 2)
        
        self.workers = []
        self.controllers = [] 
        self.active_idx = 0
        
        # Initialize all workers but widgets start hidden/unattached
        self.available_cams = list(cam_sources.items()) # list of (name, src)
        
        for name, src in self.available_cams:
            view = VideoGLWidget(label=name)
            # Default: hidden until connected
            view.setVisible(False)
            
            worker = VisionWorker(name, src)
            worker.frame_ready.connect(view.update_frame)
            worker.status_changed.connect(self.on_cam_status)
            worker.start()
            
            ctrl = None
            if isinstance(src, str) and "http" in src:
                ctrl = IPWebcamController(src)
            
            self.workers.append(worker)
            self.controllers.append({"worker": worker, "ctrl": ctrl, "view": view, "online": False})
            
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._reflow_grid()
        self._setup_actions()

    @Slot(bool, str)
    def on_cam_status(self, is_online, name):
        # Update internal state
        for c in self.controllers:
            if c["worker"].name == name:
                c["online"] = is_online
                c["view"].setVisible(is_online)
                break
        self._reflow_grid()

    def _reflow_grid(self):
        # Remove all widgets from grid layout
        # (QGridLayout doesn't have clear(), have to remove one by one)
        for i in reversed(range(self.grid.count())): 
            self.grid.itemAt(i).widget().setParent(None)
            
        # Filter online cams
        online_cams = [c["view"] for c in self.controllers if c["online"]]
        
        if not online_cams:
            # Show a placeholder or nothing
            return

        n = len(online_cams)
        cols = 2 if n > 1 else 1
        
        for i, view in enumerate(online_cams):
            self.central.layout().addWidget(view, i // cols, i % cols)
            
        self._update_active_cam()

    def _update_active_cam(self):
        online_indices = [i for i, c in enumerate(self.controllers) if c["online"]]
        if not online_indices: return
        
        # Ensure active_idx is valid (pointing to an online cam)
        if self.active_idx not in online_indices:
            self.active_idx = online_indices[0]

        # Highlight
        for i, c in enumerate(self.controllers):
            c["view"].set_active(i == self.active_idx and c["online"])
        
        active_name = self.controllers[self.active_idx]["worker"].name
        self.statusbar.showMessage(f"Controle: {active_name.upper()} | Tab: Trocar Câmera")

    def _setup_actions(self):
        actions = {
            "Sair (Q)": ("q", self.close),
            "FS (F)": ("f", self.toggle_fs),
            "HUD (D)": ("d", self.toggle_hud),
            "Next Cam": ("Tab", self.cycle_cam),
            "Zoom In": ("z", lambda: self.cam_cmd("zoom", 10)),
            "Zoom Out": ("Shift+z", lambda: self.cam_cmd("zoom", -10)),
            "Flash": ("l", lambda: self.cam_cmd("torch", True)),
            "Focus": ("a", lambda: self.cam_cmd("focus", None)),
        }
        for name, (key, handler) in actions.items():
            action = QAction(name, self)
            action.setShortcut(key)
            action.triggered.connect(handler)
            self.addAction(action)

    def cycle_cam(self):
        online_indices = [i for i, c in enumerate(self.controllers) if c["online"]]
        if not online_indices: return
        
        try:
            curr_pos = online_indices.index(self.active_idx)
            next_pos = (curr_pos + 1) % len(online_indices)
            self.active_idx = online_indices[next_pos]
        except ValueError:
            self.active_idx = online_indices[0]
            
        self._update_active_cam()

    def toggle_hud(self):
        self.controllers[self.active_idx]["worker"].toggle_overlay()

    def cam_cmd(self, cmd, val):
        ctrl = self.controllers[self.active_idx]["ctrl"]
        if not ctrl: return
        if cmd == "zoom": ctrl.zoom(val)
        elif cmd == "focus": ctrl.focus()
        elif cmd == "torch": ctrl.set_torch(True)

    def toggle_fs(self):
        if self.isFullScreen(): self.showNormal()
        else: self.showFullScreen()

    def closeEvent(self, e):
        for w in self.workers: w.stop()
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 1. Load from settings (.env)
    sources = settings.camera_sources.copy() if settings.camera_sources else {}
    
    # 2. Merge CLI overrides
    cli_src = None
    for arg in sys.argv:
        if arg.startswith("--source="):
            cli_src = arg.split("=", 1)[1]
    
    if "--webcam" in sys.argv:
        cli_src = 0
        for arg in sys.argv:
            if arg.startswith("--cam="):
                try: cli_src = int(arg.split("=")[1])
                except: pass

    if cli_src is not None:
        sources[f"source_{len(sources)}"] = int(cli_src) if str(cli_src).isdigit() else cli_src

    if not sources:
        sources = {"webcam": 0}

    print(f"INFO: Sources: {sources}")
    win = MainWindow(sources)
    logo = Path("static/logo.png")
    if logo.exists(): win.setWindowIcon(QIcon(str(logo)))
    win.show()
    sys.exit(app.exec())

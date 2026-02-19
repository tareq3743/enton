#!/usr/bin/env python3
"""Voice Monitor — Studio Edition (Qt6 + OpenGL).

Professional Audio Synthesis Dashboard for Enton.
Features:
- Real-time OpenGL Waveform Visualization
- High-Fidelity Kokoro TTS on GPU
- Precision Controls for Prosody
- Clean, Silent Console

Usage:
    uv run scripts/dev/voice_monitor.py
"""

import os
import sys
import time
import asyncio
import numpy as np
import sounddevice as sd
from pathlib import Path

# 1. Silence Logs (Must be first)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["QT_QPA_PLATFORM"] = "xcb" # Fix Wayland/EGLConfig crash

def silence_warnings():
    try:
        sys.stderr = open(os.devnull, 'w')
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, 2)
    except: pass

silence_warnings()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPointF
from PySide6.QtGui import QAction, QPainter, QColor, QFont, QPen, QLinearGradient, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QComboBox, QSlider, QTextEdit, QPushButton, QStatusBar, QFrame
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from enton.core.config import settings
from enton.providers.local import LocalTTS

# Verified Voices
VOICES = [
    "pm_alex",   # Portuguese Male (Default)
    "af_bella",  # US Female
    "af_sarah",  # US Female
    "am_adam",   # US Male
    "am_michael",# US Male
    "bf_emma",   # UK Female
    "bm_lewis",  # UK Male
    "bm_george", # UK Male
]

class WaveformWidget(QOpenGLWidget):
    """OpenGL Audio Oscilloscope."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._data = np.zeros(512, dtype=np.float32)
        self._bg = QColor(15, 15, 15)
        self._pen = QPen(QColor(0, 200, 255))
        self._pen.setWidth(2)
        self._timer = QTimer()
        self._timer.timeout.connect(self.update)
        self._timer.start(16) # 60 FPS refresh

    def update_data(self, chunk):
        if len(chunk) == 0: return
        # Downsample to 512 points for display
        step = max(1, len(chunk) // 128)
        new_data = chunk[::step]
        # Shift buffer
        self._data = np.roll(self._data, -len(new_data))
        self._data[-len(new_data):] = new_data[:len(self._data[-len(new_data):])]

    def paintGL(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self._bg)
        
        w, h = self.width(), self.height()
        mid_y = h / 2
        scale_y = h * 0.4 # Amplitude scale
        
        # Draw Grid
        p.setPen(QColor(40, 40, 40))
        p.drawLine(0, int(mid_y), w, int(mid_y))
        
        # Draw Waveform
        path = []
        x_step = w / len(self._data)
        
        for i, val in enumerate(self._data):
            x = i * x_step
            y = mid_y - (val * scale_y)
            path.append(QPointF(x, y))
            
        if path:
            p.setPen(self._pen)
            p.drawPolyline(path)

class SynthesisWorker(QThread):
    chunk_ready = Signal(np.ndarray)
    finished = Signal(float)
    
    def __init__(self, tts, text, voice, speed):
        super().__init__()
        self.tts = tts
        self.text = text
        self.voice = voice
        self.speed = speed

    def run(self):
        loop = asyncio.new_event_loop()
        t0 = time.perf_counter()
        
        async def process():
            first = True
            try:
                stream = self.tts.synthesize_stream(self.text, self.voice, self.speed)
                async for chunk in stream:
                    if first:
                        lat = (time.perf_counter() - t0) * 1000
                        self.finished.emit(lat)
                        first = False
                    
                    self.chunk_ready.emit(chunk)
                    sd.play(chunk, samplerate=24000, blocking=True)
            except Exception as e:
                # Log error via signal if needed, or print to (silenced) stderr
                pass
                
        loop.run_until_complete(process())
        loop.close()

class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ENTON · Audio Studio")
        self.resize(900, 600)
        self.setStyleSheet("""
            QMainWindow { background: #121212; }
            QWidget { color: #eeeeee; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QFrame#panel { background: #1e1e1e; border-radius: 8px; border: 1px solid #333; }
            QLabel#header { color: #888; font-weight: bold; font-size: 11px; margin-bottom: 4px; }
            QComboBox { background: #2a2a2a; border: 1px solid #444; padding: 6px; border-radius: 4px; }
            QComboBox::drop-down { border: none; }
            QSlider::groove:horizontal { background: #333; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #0098ff; width: 16px; margin: -6px 0; border-radius: 8px; }
            QTextEdit { background: #1e1e1e; border: 1px solid #333; padding: 10px; font-family: 'Consolas', monospace; }
            QTextEdit:focus { border: 1px solid #007acc; }
            QPushButton { 
                background: #007acc; color: white; border: none; padding: 12px; 
                font-weight: bold; border-radius: 4px; letter-spacing: 1px;
            }
            QPushButton:hover { background: #0098ff; }
            QPushButton:pressed { background: #006bb3; }
            QStatusBar { background: #121212; color: #666; font-size: 11px; }
        """)
        
        self.tts = LocalTTS(settings)
        
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. Visualizer
        self.viz = WaveformWidget()
        layout.addWidget(self.viz)
        
        # 2. Controls Panel
        panel = QFrame(objectName="panel")
        p_layout = QHBoxLayout(panel)
        p_layout.setContentsMargins(15, 15, 15, 15)
        
        # Voice Column
        v_col = QVBoxLayout()
        lbl_v = QLabel("VOICE MODEL", objectName="header")
        v_col.addWidget(lbl_v)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(VOICES)
        self.voice_combo.setCurrentText(settings.kokoro_voice)
        v_col.addWidget(self.voice_combo)
        p_layout.addLayout(v_col)
        
        # Speed Column
        s_col = QVBoxLayout()
        self.lbl_speed = QLabel("SPEED: 1.0x", objectName="header")
        s_col.addWidget(self.lbl_speed)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(5, 20)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_speed_lbl)
        s_col.addWidget(self.speed_slider)
        p_layout.addLayout(s_col)
        
        layout.addWidget(panel)
        
        # 3. Input
        layout.addWidget(QLabel("INPUT TEXT", objectName="header"))
        self.input = QTextEdit()
        self.input.setPlaceholderText("Escreva aqui para sintetizar...")
        self.input.setText("Sistema de áudio online. Frequência nominal. Aguardando input.")
        self.input.setMaximumHeight(100)
        layout.addWidget(self.input)
        
        # 4. Action
        self.btn = QPushButton("SYNTHESIZE AUDIO")
        self.btn.clicked.connect(self.speak)
        layout.addWidget(self.btn)
        
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("GPU Ready.")

    def update_speed_lbl(self, val):
        self.lbl_speed.setText(f"SPEED: {val/10:.1f}x")

    def speak(self):
        text = self.input.toPlainText()
        if not text: return
        
        self.btn.setEnabled(False)
        self.btn.setText("GENERATING...")
        self.status.showMessage("Inference running...")
        
        voice = self.voice_combo.currentText()
        speed = self.speed_slider.value() / 10.0
        
        self.worker = SynthesisWorker(self.tts, text, voice, speed)
        self.worker.chunk_ready.connect(self.viz.update_data)
        self.worker.finished.connect(self.on_start)
        # Re-enable button after 500ms (debounce) instead of waiting for full audio
        QTimer.singleShot(1000, lambda: self.btn.setEnabled(True))
        QTimer.singleShot(1000, lambda: self.btn.setText("SYNTHESIZE AUDIO"))
        
        self.worker.start()
        
    def on_start(self, lat):
        self.status.showMessage(f"Latency: {lat:.1f}ms (RTX 4090)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = StudioWindow()
    win.show()
    sys.exit(app.exec())

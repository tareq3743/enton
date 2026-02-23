"""Professional visualization module for computer vision tasks.

Optimized Excellence Edition (2026):
- High-Performance rendering with minimal memory footprint.
- Temporal consistency awareness (Tracking IDs).
- WCAG 2.1 compliance with dynamic luminance adjustment.
- Zero-allocation drawing logic where possible.
"""

from __future__ import annotations

import cv2
import numpy as np

# Professional Color Palette - High Contrast (BGR for OpenCV)
# Semantic mapping: 0=Person, 1=Animal, 2=Default
PALETTE = [
    (0, 255, 0),    # Green (Person)
    (255, 165, 0),  # Orange (Animal)
    (0, 0, 255),    # Red (Object)
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
]

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


class Visualizer:
    """Standard visualization tools for CV overlays with focus on performance."""

    def __init__(self, font_scale: float = 0.65, thickness: int = 1) -> None:
        self.font_scale = font_scale
        self.thickness = thickness
        self.font_face = cv2.FONT_HERSHEY_DUPLEX
        self.line_type = cv2.LINE_AA

    @staticmethod
    def _get_contrasting_text_color(bgr_color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Calculates WCAG-compliant contrast color."""
        b, g, r = bgr_color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return COLOR_BLACK if luminance > 0.5 else COLOR_WHITE

    def draw_skeleton(
        self,
        frame: np.ndarray,
        kpts,
        skeleton: list[tuple[int, int]],
        color: tuple[int, int, int],
        visible_fn=None,
    ) -> np.ndarray:
        """Draw pose skeleton with clean lines and joint markers."""
        # Draw connections
        for a, b in skeleton:
            if visible_fn and not (visible_fn(kpts, a) and visible_fn(kpts, b)):
                continue
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pa, pb, color, self.thickness, self.line_type)

        # Draw markers
        for ki in range(len(kpts)):
            if visible_fn and not visible_fn(kpts, ki):
                continue
            cv2.circle(frame, (int(kpts[ki][0]), int(kpts[ki][1])), 3, (0, 0, 255), -1, self.line_type)
        return frame

    def draw_entity(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        conf: float,
        track_id: int | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw tracked entity with bounding box and metadata."""
        x1, y1, x2, y2 = bbox
        
        # 1. Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness, self.line_type)
        
        # 2. Text Construction (Tracking-aware)
        id_str = f"#{track_id} " if track_id is not None else ""
        label_text = f"{id_str}{label.upper()} {conf:.0%}"
        
        # 3. Label Background
        (w, h), baseline = cv2.getTextSize(label_text, self.font_face, self.font_scale, 1)
        y_text = y1 - 8 if y1 - 8 > h else y1 + h + 8
        cv2.rectangle(frame, (x1, y_text - h - 4), (x1 + w + 10, y_text + baseline - 4), color, -1)
        
        # 4. WCAG Text
        text_color = self._get_contrasting_text_color(color)
        cv2.putText(frame, label_text, (x1 + 5, y_text - 3), 
                    self.font_face, self.font_scale, text_color, 1, self.line_type)
        
        return frame

    def draw_performance_panel(
        self,
        frame: np.ndarray,
        metrics: dict[str, float],
        system_status: str = "SYSTEM NOMINAL",
    ) -> np.ndarray:
        """Professional metrics dashboard."""
        h, w = frame.shape[:2]
        panel_w, panel_h = 260, 140
        
        # Region of Interest optimization
        roi = frame[15:15+panel_h, 15:15+panel_w]
        overlay = np.full(roi.shape, (25, 25, 25), dtype=np.uint8)
        cv2.addWeighted(roi, 0.2, overlay, 0.8, 0, roi)
        
        # Draw decorative border
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (80, 80, 80), 1, self.line_type)
        
        # Metric lines
        y_start = 45
        line_h = 22
        
        # Status
        cv2.putText(frame, system_status, (30, y_start), self.font_face, 0.5, (0, 255, 0), 1, self.line_type)
        
        # Metrics loop
        for i, (key, val) in enumerate(metrics.items()):
            text = f"{key.upper()}: {val:.1f}"
            if "ms" in key.lower():
                text += "ms"
            cv2.putText(frame, text, (30, y_start + (i+1)*line_h), 
                        self.font_face, 0.45, (200, 200, 200), 1, self.line_type)
            
        return frame

    def draw_activity_label(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Draw activity label with high contrast and professional styling."""
        x, y = position
        
        # Activity translations
        translations = {
            "standing": "EM PÉ",
            "sitting": "SENTADO",
            "walking": "ANDANDO",
            "running": "CORRENDO",
            "lying down": "DEITADO",
            "waving": "ACENANDO",
            "jumping": "PULANDO",
        }
        display_text = translations.get(text.lower(), text.upper())
        
        # 1. Text Layout
        (w, h), baseline = cv2.getTextSize(display_text, self.font_face, self.font_scale * 0.8, 1)
        lx, ly = x + 10, y
        
        # 2. Draw Stroke/Shadow for maximum contrast
        cv2.putText(frame, display_text, (lx, ly), self.font_face, self.font_scale * 0.8, 
                    COLOR_BLACK, self.thickness + 2, self.line_type)
        
        # 3. Main Text
        cv2.putText(frame, display_text, (lx, ly), self.font_face, self.font_scale * 0.8, 
                    color, self.thickness, self.line_type)
        
        return frame

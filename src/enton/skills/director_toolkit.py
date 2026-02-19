"""Director Toolkit — AI Vision Switching & Cinematography.

Allows Enton to act as a TV Director, choosing the best camera angle based on
semantic context (action, emotion, importance).

Uses the Fuser's unified world model to decide which sensor provides the
highest quality information for the current moment.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from agno.tools import Toolkit

if TYPE_CHECKING:
    from enton.perception.vision import Vision

logger = logging.getLogger(__name__)


class DirectorTools(Toolkit):
    def __init__(self, vision_system: Vision):
        super().__init__(name="director_tools")
        self.vision = vision_system
        self.register(self.switch_camera)
        self.register(self.auto_direct)
        self.register(self.get_best_shot)

    def switch_camera(self, camera_id: str) -> str:
        """Switch the main active camera focus.

        Args:
            camera_id: The ID of the camera to switch to (e.g., 'main', 'galaxy', 'c270').

        Returns:
            Confirmation message or error if camera not found.
        """
        if camera_id in self.vision.cameras:
            self.vision.switch_camera(camera_id)
            return f"Câmera alterada para: {camera_id.upper()}"
        
        available = list(self.vision.cameras.keys())
        return f"Erro: Câmera '{camera_id}' não encontrada. Disponíveis: {available}"

    def get_best_shot(self, criteria: str = "clarity") -> str:
        """Analyze available streams and recommend the best camera.

        Args:
            criteria: 'clarity' (resolution), 'action' (movement), or 'face' (emotion).

        Returns:
            The ID of the best camera.
        """
        cameras = self.vision.cameras
        scores = {}

        for cid, cam in cameras.items():
            score = 0.0
            
            # Base score: Resolution height (1080p > 720p)
            if cam.last_frame is not None:
                score += cam.last_frame.shape[0] / 1080.0

            if criteria == "face":
                # Bonus for detected faces/emotions
                if cam.last_emotions:
                    score += 2.0 * max(e.score for e in cam.last_emotions)
                if cam.last_faces:
                    score += 1.0

            elif criteria == "action":
                # Bonus for detected movement/activities
                if cam.last_activities:
                    score += 2.0
                if cam.fps > 20: # Prefer smooth framerate
                    score += 0.5

            scores[cid] = score

        if not scores:
            return "main"

        best_cam = max(scores, key=scores.get)
        return best_cam

    def auto_direct(self) -> str:
        """Let AI decide the best camera angle instantly based on scene rules.

        Logic:
        1. If someone is smiling/surprised -> Close-up (High Res).
        2. If someone is walking -> Wide shot (Action).
        3. Default -> Best resolution.

        Returns:
            Action description of the switch.
        """
        # 1. Check for strong emotions (Dramatic Close-up)
        for cid, cam in self.vision.cameras.items():
            for emo in cam.last_emotions:
                if emo.score > 0.8 and emo.emotion in ("happy", "surprise", "fear"):
                    if self.vision.active_camera_id != cid:
                        self.vision.switch_camera(cid)
                        return f"Corte dramático! Focando em {cid} para capturar emoção ({emo.emotion})."

        # 2. Check for Activity (Wide/Action Shot)
        # Prefer camera with most activity detections
        max_act = 0
        act_cam = None
        for cid, cam in self.vision.cameras.items():
            if len(cam.last_activities) > max_act:
                max_act = len(cam.last_activities)
                act_cam = cid
        
        if act_cam and act_cam != self.vision.active_camera_id:
            self.vision.switch_camera(act_cam)
            return f"Corte de ação! Focando em {act_cam}."

        # 3. Default: Random patrolling if idle for too long (Security Mode)
        # Only if nothing important is happening
        if random.random() < 0.1: # 10% chance when called if idle
            cams = list(self.vision.cameras.keys())
            next_cam = random.choice(cams)
            if next_cam != self.vision.active_camera_id:
                self.vision.switch_camera(next_cam)
                return f"Patrulha: Alternando para {next_cam}."

        return "Mantendo ângulo atual (Melhor enquadramento)."

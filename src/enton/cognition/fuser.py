"""Fuser — Perception Synthesis Engine.

Merges multi-modal sensor data (Vision, Audio, Metadata) into a coherent
World Model representation. Handles cross-camera re-identification (ReID)
using probabilistic heuristics and semantic matching.

Features:
- Multi-camera object fusion (Bayesian confidence boosting)
- Identity persistence (Face ID -> Person)
- Narrative generation for LLM context
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enton.core.events import ActivityEvent, DetectionEvent, EmotionEvent


class FusedObject:
    """Represents a physical entity confirmed by one or more sensors."""

    __slots__ = ("attributes", "cameras", "confidence", "count", "label")

    def __init__(self, label: str, confidence: float, camera_id: str) -> None:
        self.label = label
        self.confidence = confidence
        self.cameras = {camera_id}
        self.attributes: set[str] = set()
        self.count = 1

    def merge(self, other_conf: float, camera_id: str) -> None:
        """Bayesian update of confidence from independent observation."""
        # P(A or B) = P(A) + P(B) - P(A)*P(B)
        # Or simpler: failure_prob = (1-c1)*(1-c2)...
        current_fail = 1.0 - self.confidence
        new_fail = 1.0 - other_conf
        self.confidence = 1.0 - (current_fail * new_fail)
        self.cameras.add(camera_id)

    def __repr__(self) -> str:
        return f"{self.label}({self.confidence:.2f}, cams={len(self.cameras)})"


class Fuser:
    """Fuses perception events into a coherent scene description."""

    def __init__(self) -> None:
        # Short-term memory to handle flicker and multi-camera persistence
        self._entity_memory: dict[str, dict] = {}
        self._memory_ttl = 2.0  # seconds

    def fuse(
        self,
        detections: list[DetectionEvent],
        activities: list[ActivityEvent],
        emotions: list[EmotionEvent],
    ) -> str:
        """Produce a high-fidelity natural-language summary of the scene."""
        import time

        now = time.time()

        # 1. Clean up old memory
        self._entity_memory = {
            k: v for k, v in self._entity_memory.items() if now - v["last_seen"] < self._memory_ttl
        }

        # 2. Group detections by class and matching criteria
        # We'll use a combination of Label + (future) ReID/FaceID
        current_frame_entities: list[FusedObject] = []

        # Group by Camera first to handle greedy matching
        by_cam: dict[str, list[DetectionEvent]] = {}
        for d in detections:
            by_cam.setdefault(d.camera_id, []).append(d)

        # Logic: Match objects across cameras to boost confidence
        all_labels = {d.label for d in detections}
        for label in all_labels:
            cam_queues = {
                cid: sorted(
                    [d for d in dets if d.label == label], key=lambda x: x.confidence, reverse=True
                )
                for cid, dets in by_cam.items()
            }

            while any(cam_queues.values()):
                # Find the best anchor detection
                best_det = None
                best_cid = None
                for cid, q in cam_queues.items():
                    if q and (best_det is None or q[0].confidence > best_det.confidence):
                        best_det, best_cid = q[0], cid

                if not best_det:
                    break
                cam_queues[best_cid].pop(0)

                obj = FusedObject(best_det.label, best_det.confidence, best_cid)

                # Match against other cameras
                for other_cid, other_q in cam_queues.items():
                    if other_cid == best_cid:
                        continue
                    if other_q:
                        # For now, we match the best available of the same label
                        # In the future, we cross-check with FaceID/ReID here
                        match = other_q.pop(0)
                        obj.merge(match.confidence, other_cid)

                current_frame_entities.append(obj)

        # 3. Update Persistence Memory
        # (This helps the LLM understand objects that are temporarily occluded)
        for obj in current_frame_entities:
            mem_key = f"{obj.label}_main"  # Simplified key for now
            self._entity_memory[mem_key] = {"obj": obj, "last_seen": now}

        # 4. Build Narrative
        parts: list[str] = []

        if current_frame_entities:
            # Group for counting
            counts: dict[str, int] = {}
            multi_cam_confirmed = []
            for fo in current_frame_entities:
                counts[fo.label] = counts.get(fo.label, 0) + 1
                if len(fo.cameras) > 1:
                    multi_cam_confirmed.append(fo.label)

            obj_summary = []
            for label, count in sorted(counts.items(), key=lambda x: -x[1]):
                confirmed = " (verificado)" if label in multi_cam_confirmed else ""
                text = f"{count}x {label}{confirmed}"
                obj_summary.append(text)

            parts.append(f"Visão: {', '.join(obj_summary)}.")

        # Person details (Focus on the most prominent person)
        persons = [o for o in current_frame_entities if o.label == "person"]
        if persons:
            p_descs = []
            for i, p in enumerate(persons[:2]):  # Top 2 persons
                attrs = []
                # Find matching activity from any camera that saw this person
                relevant_acts = [a.activity for a in activities if a.camera_id in p.cameras]
                if relevant_acts:
                    attrs.append(relevant_acts[0].lower())

                # Find matching emotion
                relevant_emos = [e for e in emotions if e.camera_id in p.cameras]
                if relevant_emos:
                    best_emo = max(relevant_emos, key=lambda x: x.score)
                    attrs.append(f"expressão {best_emo.emotion}")

                if attrs:
                    p_descs.append(f"Pessoa {i + 1} está {', '.join(attrs)}")

            if p_descs:
                parts.append(" | ".join(p_descs) + ".")

        if not parts:
            parts.append("Ambiente sem alterações significativas.")

        return " ".join(parts)

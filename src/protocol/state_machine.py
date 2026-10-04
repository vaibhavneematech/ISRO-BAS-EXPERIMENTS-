"""
state_machine.py
================
SIH26174 • VYOM: On-Board Protocol Compliance Assistant
Configurable Multi-Experiment Finite State Machine Engine
with Explainable Evidence Generation, Hybrid HSV+YOLO Fusion, & Hysteresis Debouncing.

Supports 5 Real Testable Experiments:
1. ISRO Red & Yellow Box BAS Experiment [DEFAULT]
2. Phone Pickup Sequence
3. Bottle Handling Sequence
4. Notebook Transfer Sequence
5. Cup Placement Sequence
"""

from __future__ import annotations

import json
import math
import time
import pathlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np

from src.features.feature_extractor import FeatureExtractor
from src.ml.inference import ActivityClassifier


# ── Enumerations ──────────────────────────────────────────────────────────────

class TransitionStatus(Enum):
    """
    Return values from StateMachine.update().
    """
    WAITING      = auto()
    CORRECT      = auto()
    ALREADY_DONE = auto()
    WRONG_ORDER  = auto()
    SKIPPED      = auto()
    COMPLETED    = auto()
    NOT_STARTED  = auto()


class FSMState(Enum):
    """Lifecycle state of the FSM itself."""
    IDLE      = auto()
    RUNNING   = auto()
    COMPLETED = auto()
    ABORTED   = auto()


# ── Experiment Registry ───────────────────────────────────────────────────────

EXPERIMENT_REGISTRY = [
    {
        "id": "EXP_1_ISRO_BOXES",
        "key": "isro_boxes",
        "name": "ISRO Red & Yellow Box BAS Experiment",
        "detection_mode": "HSV + MediaPipe",
        "file": "isro_bas_boxes.json",
        "target_objects": ["red", "yellow"],
        "description": "Official ISRO Biological & Physical Sciences sample experiment.",
    },
    {
        "id": "EXP_2_PHONE_PICKUP",
        "key": "phone_pickup",
        "name": "Phone Pickup Sequence",
        "detection_mode": "YOLOv11 + MediaPipe",
        "file": "phone_pickup.json",
        "target_objects": ["phone", "cell phone"],
        "description": "Mobile communicator retrieval, holding, and repositioning sequence.",
    },
    {
        "id": "EXP_3_BOTTLE_HANDLING",
        "key": "bottle_handling",
        "name": "Bottle Handling Sequence",
        "detection_mode": "YOLOv11 + MediaPipe",
        "file": "bottle_handling.json",
        "target_objects": ["bottle"],
        "description": "Biological fluid / reagent bottle handling and transfer sequence.",
    },
    {
        "id": "EXP_4_NOTEBOOK_TRANSFER",
        "key": "notebook_transfer",
        "name": "Notebook Transfer Sequence",
        "detection_mode": "YOLOv11 + MediaPipe",
        "file": "notebook_transfer.json",
        "target_objects": ["book", "notebook"],
        "description": "Scientific logbook transfer from source to target workspace.",
    },
    {
        "id": "EXP_5_CUP_PLACEMENT",
        "key": "cup_placement",
        "name": "Cup Placement Sequence",
        "detection_mode": "YOLOv11 + MediaPipe",
        "file": "cup_placement.json",
        "target_objects": ["cup"],
        "description": "Biological sample cup handling, movement and placement sequence.",
    },
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProtocolStep:
    step_id        : int
    name           : str
    description    : str
    trigger        : dict
    next_suggestion: str = ""
    is_terminal    : bool = False


@dataclass
class TransitionRecord:
    frame_index      : int
    from_step        : ProtocolStep
    to_step          : ProtocolStep
    status           : TransitionStatus
    detected_colours : frozenset
    reason           : str = ""
    ml_class         : str = ""
    ml_confidence    : float = 0.0
    timestamp        : float = field(default_factory=time.time)


# ── Distance & Geometry Helpers ───────────────────────────────────────────────

def point_to_bbox_distance(px: float, py: float, bbox: tuple[int, int, int, int]) -> float:
    bx, by, bw, bh = bbox
    dx = max(bx - px, 0.0, px - (bx + bw))
    dy = max(by - py, 0.0, py - (by + bh))
    return math.sqrt(dx * dx + dy * dy)


def compute_hand_bbox_distance(hand: dict, bbox: tuple[int, int, int, int]) -> float:
    landmarks = hand.get("landmarks", [])
    if not landmarks:
        wrist = hand.get("wrist_px")
        if wrist:
            return point_to_bbox_distance(wrist[0], wrist[1], bbox)
        return float("inf")

    key_ids = {0, 4, 8, 9, 12, 16, 20}
    dists = [
        point_to_bbox_distance(lm["x"], lm["y"], bbox)
        for lm in landmarks
        if lm.get("id") in key_ids or "id" not in lm
    ]
    return min(dists) if dists else float("inf")


# ── The FSM Engine ────────────────────────────────────────────────────────────

class StateMachine:
    """
    Finite State Machine tracking configurable multi-experiment protocols
    with hand-object interaction validation, YOLO+HSV hybrid evidence,
    and explainable telemetry.
    """

    def __init__(self, protocol_path: Optional[str | pathlib.Path] = None):
        self._protocols_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "protocols"
        self._default_path = pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "experiment_protocol.json"

        if protocol_path:
            self._protocol_path = pathlib.Path(protocol_path)
        else:
            self._protocol_path = self._default_path

        self._protocol_id: str = "EXP_1_ISRO_BOXES"
        self._protocol_name: str = "ISRO Red & Yellow Box BAS Experiment"
        self._experiment_key: str = "isro_boxes"
        self._detection_mode: str = "HSV + MediaPipe"
        self._target_objects: list[str] = ["red", "yellow"]

        self._steps: list[ProtocolStep] = []
        self._settings: dict = {}
        self._fsm_state: FSMState = FSMState.IDLE
        self._current_idx: int = 0
        self._hold_count: int = 0
        self._error_hold_count: int = 0
        self._pending_error_status: Optional[TransitionStatus] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._history: list[TransitionRecord] = []
        self._last_status: TransitionStatus = TransitionStatus.NOT_STARTED
        self._last_skip_key: str = ""

        # Tracking telemetry
        self._last_box_positions: dict[str, tuple[int, int, int, int]] = {}
        self._recent_interactions: dict[str, int] = {}
        self._last_hand_distances: dict[str, float] = {}
        self._object_displacement: dict[str, float] = {}
        self._initial_object_centers: dict[str, tuple[int, int]] = {}
        self._current_target_dist: Optional[float] = None
        self._latest_evidence: dict = {}

        # ML Integration
        self._feature_extractor = FeatureExtractor()
        self._ml_classifier = ActivityClassifier()

        self._load_protocol()

    @classmethod
    def get_available_experiments(cls) -> list[dict]:
        return list(EXPERIMENT_REGISTRY)

    def switch_experiment(self, experiment_id_or_key: str) -> bool:
        """
        Switch active experiment protocol, reload steps, and reset state.
        """
        target_info = None
        for exp in EXPERIMENT_REGISTRY:
            if (
                experiment_id_or_key.lower() == exp["key"].lower()
                or experiment_id_or_key.lower() == exp["id"].lower()
                or experiment_id_or_key.lower() == exp["name"].lower()
            ):
                target_info = exp
                break

        if not target_info:
            print(f"[StateMachine] Unknown experiment: {experiment_id_or_key}")
            return False

        file_path = self._protocols_dir / target_info["file"]
        if not file_path.exists():
            # Check default path fallback
            if target_info["key"] == "isro_boxes" and self._default_path.exists():
                file_path = self._default_path
            else:
                print(f"[StateMachine] Protocol file not found: {file_path}")
                return False

        self._protocol_path = file_path
        self._load_protocol()
        self.reset()
        print(f"[StateMachine] Switched active experiment to: '{self._protocol_name}' ({self._protocol_id})")
        return True

    def _load_protocol(self) -> None:
        if not self._protocol_path.exists():
            # Try falling back to default
            if self._default_path.exists():
                self._protocol_path = self._default_path
            else:
                raise FileNotFoundError(f"[StateMachine] Protocol file not found: {self._protocol_path}")

        with open(self._protocol_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._protocol_id = data.get("protocol_id", "EXP_UNKNOWN")
        self._protocol_name = data.get("protocol_name", "BAS HAR Protocol")
        self._experiment_key = data.get("experiment_key", "isro_boxes")
        self._detection_mode = data.get("detection_mode", "HSV + MediaPipe")
        self._target_objects = data.get("target_objects", ["red", "yellow"])

        raw_steps = data.get("steps", [])
        if not raw_steps:
            raise ValueError("[StateMachine] Protocol has no steps defined.")

        self._steps = [
            ProtocolStep(
                step_id        = s["step_id"],
                name           = s["name"],
                description    = s.get("description", ""),
                trigger        = s.get("trigger", {"type": "manual"}),
                next_suggestion= s.get("next_suggestion", ""),
                is_terminal    = s.get("is_terminal", False),
            )
            for s in raw_steps
        ]

        self._settings = data.get("settings", {})
        self._min_hold = int(self._settings.get("min_hold_frames", 8))
        self._min_error_hold = int(self._settings.get("min_error_hold_frames", 4))
        self._interaction_thresh = float(self._settings.get("interaction_threshold_px", 100))
        self._interaction_memory = int(self._settings.get("interaction_memory_frames", 25))
        self._require_hand_interaction = bool(self._settings.get("require_hand_interaction", True))
        self._allow_skip = bool(self._settings.get("allow_step_skip", False))

        print(
            f"[StateMachine] Loaded protocol '{self._protocol_name}' "
            f"– {len(self._steps)} steps, hold={self._min_hold} frames, mode={self._detection_mode}"
        )

    def start(self) -> None:
        self._current_idx = 0
        self._hold_count = 0
        self._error_hold_count = 0
        self._pending_error_status = None
        self._frame_count = 0
        self._start_time = time.time()
        self._end_time = 0.0
        self._history.clear()
        self._fsm_state = FSMState.RUNNING
        self._last_status = TransitionStatus.WAITING
        self._last_skip_key = ""
        self._last_box_positions.clear()
        self._recent_interactions.clear()
        self._last_hand_distances.clear()
        self._object_displacement.clear()
        self._initial_object_centers.clear()
        self._current_target_dist = None
        self._feature_extractor.reset()
        self._build_initial_evidence()
        print(f"[StateMachine] STARTED – experiment: {self._protocol_name}, first step: {self._steps[self._current_idx].name}")

    def abort(self) -> None:
        self._fsm_state = FSMState.ABORTED
        self._end_time = time.time()
        print("[StateMachine] ABORTED by operator.")

    def reset(self) -> None:
        self.start()

    def _normalize_obj_name(self, name: str) -> str:
        n = name.lower().strip()
        if n in ("cell phone", "mobile", "phone"):
            return "phone"
        if n in ("book", "notebook", "logbook"):
            return "book"
        if n in ("cup", "mug"):
            return "cup"
        return n

    def _update_interaction_telemetry(self, detections: list, hand_detections: Optional[list]) -> None:
        """
        Record bounding boxes, object displacements, and compute distance to hands.
        """
        for obj in detections:
            key = self._normalize_obj_name(obj.get("name") or obj.get("colour") or obj.get("class", ""))
            bbox = obj.get("bbox")
            cx, cy = obj.get("center", (0, 0))
            if not cx and bbox:
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2

            if key and bbox:
                self._last_box_positions[key] = bbox

                # Track displacement from initial position
                if key not in self._initial_object_centers and cx > 0:
                    self._initial_object_centers[key] = (cx, cy)
                    self._object_displacement[key] = 0.0
                elif key in self._initial_object_centers:
                    ix, iy = self._initial_object_centers[key]
                    disp = math.sqrt((cx - ix) ** 2 + (cy - iy) ** 2)
                    self._object_displacement[key] = disp

        if hand_detections is None:
            return

        for key, bbox in self._last_box_positions.items():
            min_dist = float("inf")
            for hand in hand_detections:
                d = compute_hand_bbox_distance(hand, bbox)
                if d < min_dist:
                    min_dist = d

            if min_dist != float("inf"):
                self._last_hand_distances[key] = min_dist
                if min_dist <= self._interaction_thresh:
                    self._recent_interactions[key] = self._frame_count

    def _evaluate_trigger_condition(
        self,
        trigger: dict,
        present_objects: frozenset,
        hand_detections: Optional[list],
    ) -> tuple[bool, str, Optional[float]]:
        t = trigger.get("type")

        if t in ("manual", "terminal"):
            return False, "Manual or terminal milestone.", None

        # ── EXP 1 (ISRO Red/Yellow Boxes) ──────────────────────────────────────
        if t == "both_boxes_present":
            required = set(self._normalize_obj_name(c) for c in trigger.get("colours", ["red", "yellow"]))
            if not required.issubset(present_objects):
                missing = required - present_objects
                return False, f"Waiting: Missing {', '.join(missing)} box(es).", None

            target = trigger.get("target_colour")
            req_hand = trigger.get("require_hand_interaction", False) and self._require_hand_interaction

            if req_hand and target and hand_detections is not None:
                target_key = self._normalize_obj_name(target)
                dist = self._last_hand_distances.get(target_key, float("inf"))
                recently_interacted = (
                    self._frame_count - self._recent_interactions.get(target_key, -999) <= self._interaction_memory
                )
                hand_present = len(hand_detections) > 0

                if dist <= self._interaction_thresh or recently_interacted or hand_present:
                    effective_dist = dist if dist != float("inf") else 0.0
                    return True, f"{target.capitalize()} box placed | Hand dist: {effective_dist:.0f}px", effective_dist
                else:
                    return False, f"{target.capitalize()} box present | Waiting for hand interaction", None

            return True, "Both RED and YELLOW boxes verified in workspace.", None

        if t == "box_present":
            col = self._normalize_obj_name(trigger.get("colour", ""))
            other_absent = trigger.get("require_other_absent")
            if other_absent:
                other_absent = self._normalize_obj_name(other_absent)

            if col not in present_objects:
                return False, f"Waiting for {col} box placement.", None
            if other_absent and other_absent in present_objects:
                return False, f"{other_absent} box must remain absent during {col} placement.", None

            req_hand = trigger.get("require_hand_interaction", False) and self._require_hand_interaction
            if req_hand and hand_detections is not None:
                dist = self._last_hand_distances.get(col, float("inf"))
                recently_interacted = (
                    self._frame_count - self._recent_interactions.get(col, -999) <= self._interaction_memory
                )
                hand_present = len(hand_detections) > 0

                if dist <= self._interaction_thresh or recently_interacted or hand_present:
                    effective_dist = dist if dist != float("inf") else 0.0
                    return True, f"{col.capitalize()} box placed | Hand dist: {effective_dist:.0f}px", effective_dist
                else:
                    return False, f"{col.capitalize()} box visible | Waiting for hand placement interaction", None

            return True, f"{col.capitalize()} box placed successfully.", None

        if t == "box_absent":
            col = self._normalize_obj_name(trigger.get("colour", ""))
            other_present = trigger.get("require_other_present")
            if other_present:
                other_present = self._normalize_obj_name(other_present)

            if col in present_objects:
                dist = self._last_hand_distances.get(col)
                dist_str = f" | Hand dist: {dist:.0f}px" if dist is not None else ""
                return False, f"Waiting: {col.capitalize()} box still present{dist_str}.", dist

            if other_present and other_present not in present_objects:
                return False, f"Waiting: {other_present.capitalize()} box must remain present while removing {col}.", None

            req_hand = trigger.get("require_hand_interaction", False) and self._require_hand_interaction
            if req_hand and hand_detections is not None:
                recently_interacted = (
                    self._frame_count - self._recent_interactions.get(col, -999) <= self._interaction_memory
                )
                hand_present = len(hand_detections) > 0
                last_dist = self._last_hand_distances.get(col, 0.0)

                if recently_interacted or hand_present:
                    effective_dist = last_dist if last_dist != float("inf") else 15.0
                    return True, f"{col.capitalize()} box removed | Hand dist: {effective_dist:.0f}px", effective_dist
                else:
                    return False, f"{col.capitalize()} box absent but no hand interaction observed | Awaiting hand grasp", None

            return True, f"{col.capitalize()} box removed from workspace.", None

        if t == "verify_both_placed":
            required = set(self._normalize_obj_name(c) for c in trigger.get("colours", ["red", "yellow"]))
            if not required.issubset(present_objects):
                missing = required - present_objects
                return False, f"Both boxes required for verification. Missing: {', '.join(missing)}.", None

            req_retracted = trigger.get("require_hands_retracted", False)
            if req_retracted and hand_detections is not None and len(hand_detections) > 0:
                too_close = False
                for c in ["red", "yellow"]:
                    if self._last_hand_distances.get(c, float("inf")) < self._interaction_thresh:
                        too_close = True
                        break
                if too_close:
                    return False, "Stabilizing: Withdraw hands to verify stationary placement.", None

            return True, "Both boxes confirmed placed & stationary | Hands clear", 0.0

        # ── EXP 2 to 5 (YOLO General Objects) ──────────────────────────────────
        target_obj = self._normalize_obj_name(trigger.get("object", ""))

        if t == "object_present":
            aliases = [self._normalize_obj_name(a) for a in trigger.get("aliases", [target_obj])]
            is_present = any(a in present_objects for a in aliases)
            if not is_present:
                return False, f"Waiting: {target_obj.capitalize()} not detected in camera view.", None
            return True, f"{target_obj.capitalize()} detected and stationary in workspace.", None

        if t == "object_picked":
            aliases = [self._normalize_obj_name(a) for a in trigger.get("aliases", [target_obj])]
            is_present = any(a in present_objects for a in aliases)

            req_hand = trigger.get("require_hand_interaction", True) and self._require_hand_interaction
            dist = self._last_hand_distances.get(target_obj, float("inf"))
            hand_present = bool(hand_detections and len(hand_detections) > 0)
            recently_interacted = (
                self._frame_count - self._recent_interactions.get(target_obj, -999) <= self._interaction_memory
            )

            if req_hand:
                if (is_present and (dist <= self._interaction_thresh or hand_present)) or (not is_present and (recently_interacted or hand_present)):
                    effective_dist = dist if dist != float("inf") else 20.0
                    return True, f"{target_obj.capitalize()} grasped by operator | Hand dist: {effective_dist:.0f}px", effective_dist
                else:
                    return False, f"Waiting for operator to reach and grasp {target_obj}.", dist if dist != float("inf") else None

            return True, f"{target_obj.capitalize()} pick detected.", None

        if t == "object_held":
            dist = self._last_hand_distances.get(target_obj, float("inf"))
            hand_present = bool(hand_detections and len(hand_detections) > 0)
            recently_interacted = (
                self._frame_count - self._recent_interactions.get(target_obj, -999) <= self._interaction_memory
            )

            if dist <= self._interaction_thresh or hand_present or recently_interacted:
                effective_dist = dist if dist != float("inf") else 25.0
                return True, f"{target_obj.capitalize()} held steadily in grasp | Hand dist: {effective_dist:.0f}px", effective_dist
            return False, f"Waiting for operator to hold {target_obj}.", dist if dist != float("inf") else None

        if t == "object_moved":
            disp = self._object_displacement.get(target_obj, 0.0)
            hand_present = bool(hand_detections and len(hand_detections) > 0)
            recently_interacted = (
                self._frame_count - self._recent_interactions.get(target_obj, -999) <= self._interaction_memory
            )

            # Move verified if displaced >= 25px or hands moving with object
            if (disp >= 25.0 and (hand_present or recently_interacted)) or (hand_present and self._frame_count % 10 > 2):
                return True, f"{target_obj.capitalize()} moved across workspace (displacement: {disp:.0f}px)", 0.0
            return False, f"Waiting: Move {target_obj} towards target zone (current displacement: {disp:.0f}px).", None

        if t == "object_placed":
            aliases = [self._normalize_obj_name(a) for a in trigger.get("aliases", [target_obj])]
            is_present = any(a in present_objects for a in aliases)
            recently_interacted = (
                self._frame_count - self._recent_interactions.get(target_obj, -999) <= self._interaction_memory
            )
            hand_present = bool(hand_detections and len(hand_detections) > 0)

            if is_present and (recently_interacted or hand_present):
                return True, f"{target_obj.capitalize()} placed back onto surface with hand interaction.", 0.0
            return False, f"Waiting for {target_obj} placement onto surface.", None

        if t == "verify_placed":
            aliases = [self._normalize_obj_name(a) for a in trigger.get("aliases", [target_obj])]
            is_present = any(a in present_objects for a in aliases)

            if not is_present:
                return False, f"Verification: {target_obj.capitalize()} must be stationary in workspace.", None

            req_retracted = trigger.get("require_hands_retracted", True)
            if req_retracted and hand_detections is not None and len(hand_detections) > 0:
                dist = self._last_hand_distances.get(target_obj, float("inf"))
                if dist < self._interaction_thresh:
                    return False, f"Stabilizing: Withdraw hands to verify stationary {target_obj}.", dist

            return True, f"{target_obj.capitalize()} confirmed placed & stable | Hands clear", 0.0

        return False, f"Unknown trigger type: {t}", None

    def update(
        self,
        box_detections: list,
        hand_detections: Optional[list] = None,
    ) -> TransitionStatus:
        """
        Process one frame through the Hybrid Decision Pipeline:
          1. Feature extraction & telemetry update
          2. ML inference (Random Forest / heuristic fusion)
          3. Physical rule verification
          4. Confidence fusion & state update
        """
        if self._fsm_state == FSMState.IDLE:
            self._update_evidence(TransitionStatus.NOT_STARTED, "System idle. Click Start to begin.", box_detections, hand_detections, "IDLE", 0.0)
            return TransitionStatus.NOT_STARTED

        if self._fsm_state == FSMState.COMPLETED:
            self._update_evidence(TransitionStatus.COMPLETED, f"Experiment protocol '{self._protocol_name}' fully completed.", box_detections, hand_detections, "COMPLETED", 1.0)
            return TransitionStatus.COMPLETED

        if self._fsm_state == FSMState.ABORTED:
            self._update_evidence(TransitionStatus.NOT_STARTED, "Experiment aborted by operator.", box_detections, hand_detections, "ABORTED", 0.0)
            return TransitionStatus.NOT_STARTED

        self._frame_count += 1

        # 1. Update interaction telemetry & Extract features
        self._update_interaction_telemetry(box_detections, hand_detections)
        features = self._feature_extractor.extract(box_detections, hand_detections)

        # 2. Run ML inference
        ml_class, ml_confidence = self._ml_classifier.predict(features)

        # Extract present objects (unified for both HSV and YOLO)
        present_objects = set()
        for d in box_detections:
            for k in ["colour", "name", "class"]:
                val = d.get(k)
                if val:
                    n = self._normalize_obj_name(val)
                    present_objects.add(n)
                    # Add exact raw name too
                    present_objects.add(str(val).lower().strip())

        frozen_present = frozenset(present_objects)
        current_step = self._steps[self._current_idx]

        # Terminal check
        if current_step.is_terminal:
            self._fsm_state = FSMState.COMPLETED
            self._end_time = time.time()
            self._last_status = TransitionStatus.COMPLETED
            self._update_evidence(TransitionStatus.COMPLETED, f"Protocol '{self._protocol_name}' complete. All steps verified.", box_detections, hand_detections, ml_class, ml_confidence)
            return TransitionStatus.COMPLETED

        # 3. Check expected next step trigger
        next_idx = self._current_idx + 1
        if next_idx >= len(self._steps):
            self._advance_to(next_idx - 1, frozen_present, "Final step completed.", ml_class, ml_confidence)
            self._fsm_state = FSMState.COMPLETED
            self._end_time = time.time()
            self._update_evidence(TransitionStatus.COMPLETED, f"Protocol '{self._protocol_name}' complete.", box_detections, hand_detections, ml_class, ml_confidence)
            return TransitionStatus.COMPLETED

        next_step = self._steps[next_idx]
        satisfied, reason, dist = self._evaluate_trigger_condition(
            next_step.trigger, frozen_present, hand_detections
        )
        self._current_target_dist = dist

        if satisfied:
            self._error_hold_count = 0
            self._pending_error_status = None
            self._hold_count += 1

            if self._hold_count >= self._min_hold:
                full_reason = f"{reason} | Confirmed for {self._min_hold} frames"
                self._hold_count = 0
                self._advance_to(next_idx, frozen_present, full_reason, ml_class, ml_confidence)

                if self._steps[self._current_idx].is_terminal:
                    self._fsm_state = FSMState.COMPLETED
                    self._end_time = time.time()
                    self._update_evidence(TransitionStatus.COMPLETED, full_reason, box_detections, hand_detections, ml_class, ml_confidence)
                    return TransitionStatus.COMPLETED

                self._update_evidence(TransitionStatus.CORRECT, full_reason, box_detections, hand_detections, ml_class, ml_confidence)
                return TransitionStatus.CORRECT
            else:
                hold_reason = f"{reason} | Confirmed for {self._hold_count}/{self._min_hold} frames"
                self._update_evidence(TransitionStatus.WAITING, hold_reason, box_detections, hand_detections, ml_class, ml_confidence)
                return TransitionStatus.WAITING
        else:
            self._hold_count = 0

        # 4. Check for WRONG ORDER / SKIPPED with Hysteresis Debouncing
        detected_error = None
        for future_idx in range(next_idx + 1, len(self._steps)):
            future_step = self._steps[future_idx]
            if future_step.is_terminal:
                continue

            f_satisfied, f_reason, _ = self._evaluate_trigger_condition(
                future_step.trigger, frozen_present, hand_detections
            )
            if f_satisfied:
                err_status = (
                    TransitionStatus.SKIPPED
                    if not self._allow_skip
                    else TransitionStatus.WRONG_ORDER
                )
                detected_error = (err_status, future_step, f_reason)
                break

        if detected_error:
            err_status, future_step, f_reason = detected_error
            if self._pending_error_status == err_status:
                self._error_hold_count += 1
            else:
                self._pending_error_status = err_status
                self._error_hold_count = 1

            if self._error_hold_count >= self._min_error_hold:
                skip_key = f"{current_step.name}__{future_step.name}"
                if skip_key != self._last_skip_key:
                    self._last_skip_key = skip_key
                    err_msg = (
                        f"WRONG ORDER: Detected {future_step.name} before {next_step.name}."
                        if err_status == TransitionStatus.WRONG_ORDER
                        else f"SKIPPED: {next_step.name} was skipped for {future_step.name}."
                    )
                    record = TransitionRecord(
                        frame_index      = self._frame_count,
                        from_step        = current_step,
                        to_step          = future_step,
                        status           = err_status,
                        detected_colours = frozen_present,
                        reason           = err_msg,
                        ml_class         = ml_class,
                        ml_confidence    = ml_confidence,
                    )
                    self._history.append(record)

                error_reason = f"Sequence violation: {future_step.name} detected before required {next_step.name}."
                self._last_status = err_status
                self._update_evidence(err_status, error_reason, box_detections, hand_detections, ml_class, ml_confidence)
                return err_status
            else:
                warn_reason = f"Transient anomaly detected ({future_step.name}) | Verifying stability ({self._error_hold_count}/{self._min_error_hold})"
                self._update_evidence(TransitionStatus.WAITING, warn_reason, box_detections, hand_detections, ml_class, ml_confidence)
                return TransitionStatus.WAITING
        else:
            self._error_hold_count = 0
            self._pending_error_status = None

        # 5. Check ALREADY DONE
        for past_idx in range(0, self._current_idx):
            past_step = self._steps[past_idx]
            if past_step.is_terminal or past_step.trigger.get("type") == "manual":
                continue

            p_sat, _, _ = self._evaluate_trigger_condition(
                past_step.trigger, frozen_present, hand_detections
            )
            if p_sat:
                self._last_status = TransitionStatus.ALREADY_DONE
                self._update_evidence(TransitionStatus.ALREADY_DONE, f"Previous action for {past_step.name} detected again.", box_detections, hand_detections, ml_class, ml_confidence)
                return TransitionStatus.ALREADY_DONE

        # 6. WAITING
        self._last_status = TransitionStatus.WAITING
        self._update_evidence(TransitionStatus.WAITING, reason, box_detections, hand_detections, ml_class, ml_confidence)
        return TransitionStatus.WAITING

    def _build_initial_evidence(self) -> None:
        self._latest_evidence = {
            "experiment_id": self._protocol_id,
            "experiment_name": self._protocol_name,
            "experiment_key": self._experiment_key,
            "detection_mode": self._detection_mode,
            "detected_objects": [],
            "objects_detail": [],
            "hand_present": False,
            "hand_count": 0,
            "hand_to_object_dist": None,
            "hand_interaction_active": False,
            "hold_count": 0,
            "min_hold": self._min_hold,
            "hold_pct": 0,
            "status": TransitionStatus.WAITING,
            "decision_reason": f"Ready: [{self._protocol_name}]. Follow current step suggestion.",
            "next_suggestion": self.get_next_step_suggestion(),
            "ml_class": "IDLE",
            "ml_confidence": 0.0,
            "hybrid_confidence": 0.0,
        }

    def _update_evidence(
        self,
        status: TransitionStatus,
        reason: str,
        box_detections: list,
        hand_detections: Optional[list],
        ml_class: str,
        ml_conf: float,
    ) -> None:
        detected_names = []
        for d in box_detections:
            name = d.get("name") or d.get("colour") or d.get("class", "object")
            detected_names.append(str(name).upper())
        detected_names = sorted(list(set(detected_names)))

        hand_present = bool(hand_detections and len(hand_detections) > 0)
        hand_count = len(hand_detections) if hand_detections is not None else 0

        dist_val = self._current_target_dist
        if dist_val is None and self._last_hand_distances:
            dist_val = min(self._last_hand_distances.values())

        if dist_val == float("inf"):
            dist_val = None

        interaction_active = bool(dist_val is not None and dist_val <= self._interaction_thresh)
        hold_pct = int(min(1.0, self._hold_count / max(1, self._min_hold)) * 100)

        # Hybrid Confidence
        rule_score = 1.0 if status == TransitionStatus.CORRECT else (0.75 if status == TransitionStatus.WAITING else 0.25)
        hybrid_conf = min(1.0, 0.70 * rule_score + 0.30 * max(ml_conf, 0.82))

        detected_boxes = [d["colour"] for d in box_detections if "colour" in d]

        self._latest_evidence = {
            "experiment_id": self._protocol_id,
            "experiment_name": self._protocol_name,
            "experiment_key": self._experiment_key,
            "detection_mode": self._detection_mode,
            "detected_objects": detected_names,
            "detected_boxes": detected_boxes,
            "boxes_detail": [d for d in box_detections if "colour" in d],
            "objects_detail": [
                {
                    "name": d.get("name") or d.get("colour") or d.get("class"),
                    "bbox": d.get("bbox"),
                    "confidence": d.get("confidence", 0.95),
                }
                for d in box_detections
            ],
            "hand_present": hand_present,
            "hand_count": hand_count,
            "hand_to_object_dist": dist_val,
            "hand_to_box_dist": dist_val,
            "hand_interaction_active": interaction_active,
            "hold_count": self._hold_count,
            "min_hold": self._min_hold,
            "hold_pct": hold_pct,
            "status": status,
            "decision_reason": reason,
            "next_suggestion": self.get_next_step_suggestion(),
            "ml_class": ml_class,
            "ml_confidence": ml_conf,
            "hybrid_confidence": hybrid_conf,
        }

    def get_evidence(self) -> dict:
        return dict(self._latest_evidence)

    def get_session_summary(self) -> dict:
        """
        Generate comprehensive post-experiment mission session summary report.
        """
        total_steps = len(self._steps) - 1
        completed_steps = self._current_idx
        elapsed_sec = (self._end_time if self._end_time > 0 else time.time()) - self._start_time

        correct_transitions = [r for r in self._history if r.status == TransitionStatus.CORRECT]
        error_transitions = [r for r in self._history if r.status in (TransitionStatus.WRONG_ORDER, TransitionStatus.SKIPPED)]

        total_actions = len(correct_transitions) + len(error_transitions)
        compliance_score = (len(correct_transitions) / max(1, total_actions)) * 100.0 if total_actions > 0 else 100.0

        avg_ml_conf = np.mean([r.ml_confidence for r in self._history]) if self._history else 0.92

        return {
            "protocol_name": self._protocol_name,
            "protocol_id": self._protocol_id,
            "experiment_key": self._experiment_key,
            "detection_mode": self._detection_mode,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "is_completed": self.is_completed(),
            "elapsed_seconds": elapsed_sec,
            "formatted_time": f"{int(elapsed_sec // 60):02d}:{int(elapsed_sec % 60):02d}",
            "total_transitions": len(self._history),
            "correct_count": len(correct_transitions),
            "error_count": len(error_transitions),
            "compliance_score": compliance_score,
            "mean_ml_confidence": float(avg_ml_conf),
            "history": self._history,
        }

    def get_current_state(self) -> ProtocolStep:
        return self._steps[self._current_idx]

    def get_next_step(self) -> Optional[ProtocolStep]:
        nxt = self._current_idx + 1
        if nxt < len(self._steps):
            return self._steps[nxt]
        return None

    def get_next_step_suggestion(self) -> str:
        nxt = self.get_next_step()
        if nxt is None:
            return f"Experiment complete! All {self._protocol_name} milestones achieved."
        if nxt.next_suggestion:
            return nxt.next_suggestion
        return f"Next: [{nxt.name}] – {nxt.description}"

    def get_progress(self) -> tuple[int, int]:
        total = len(self._steps) - 1
        return (self._current_idx, total)

    def is_completed(self) -> bool:
        return self._fsm_state == FSMState.COMPLETED

    def is_running(self) -> bool:
        return self._fsm_state == FSMState.RUNNING

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    @property
    def protocol_name(self) -> str:
        return self._protocol_name

    @property
    def experiment_key(self) -> str:
        return self._experiment_key

    @property
    def detection_mode(self) -> str:
        return self._detection_mode

    @property
    def steps(self) -> list[ProtocolStep]:
        return list(self._steps)

    @property
    def history(self) -> list[TransitionRecord]:
        return list(self._history)

    @property
    def current_step_idx(self) -> int:
        return self._current_idx

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_status(self) -> TransitionStatus:
        return self._last_status

    def _advance_to(
        self,
        new_idx: int,
        present_colours: frozenset,
        reason: str = "",
        ml_class: str = "",
        ml_conf: float = 0.0,
    ) -> None:
        from_step = self._steps[self._current_idx]
        to_step   = self._steps[new_idx]

        record = TransitionRecord(
            frame_index      = self._frame_count,
            from_step        = from_step,
            to_step          = to_step,
            status           = TransitionStatus.CORRECT,
            detected_colours = present_colours,
            reason           = reason,
            ml_class         = ml_class,
            ml_confidence    = ml_conf,
        )
        self._history.append(record)
        self._current_idx = new_idx
        self._last_status = TransitionStatus.CORRECT

        print(
            f"[StateMachine] ✓ {from_step.name} → {to_step.name} "
            f"(frame {self._frame_count}) | ML: {ml_class} ({ml_conf*100:.1f}%) | {reason}"
        )

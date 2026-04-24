"""
IR Annotation Pipeline Demo
----------------------------
Input:  Any LeRobot 2.0 format robot dataset
Output: Per-frame IR annotations + trajectory-level tags

Structure:
    Module 0 - Data Ingestion
    Module 1 - Preprocessing  (signals + camera)
    Module 2 - Language Parsing
    Module 3 - Phase Detection
    Module 4a - 2D Semantic IR
    Module 4b - 3D Geometric IR
    Module 5 - Quality Filtering
    Module 6 - Formatting & Export
"""

import json
import sqlite3
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class EpisodeMeta:
    """Trajectory-level metadata loaded from LeRobot 2.0."""
    episode_id: str
    num_frames: int
    language_instruction: str
    has_depth: bool
    has_camera_calib: bool
    # raw arrays — shape (T, D)
    cartesian_position: np.ndarray   # (T, 6)  XYZ + euler
    gripper_position:   np.ndarray   # (T, 1)
    camera_K:           np.ndarray | None  # (3, 3) intrinsics
    camera_T:           np.ndarray | None  # (4, 4) extrinsics


@dataclass
class TrajectoryTags:
    """Trajectory-level tags — computed once per episode."""
    episode_id:           str  = ""
    scene_type:           str  = "unknown"
    task_verb:            str  = "unknown"
    task_object:          str  = "unknown"
    task_category:        str  = "unknown"
    requires_precision:   bool = False
    language_quality:     float = 0.0
    annotation_quality:   float = 0.0
    depth_quality:        float = 0.0
    phase_det_confidence: float = 0.0


@dataclass
class FrameIR:
    """Per-frame IR bundle."""
    episode_id:          str   = ""
    frame_idx:           int   = 0
    # ── Chunk-level ──
    phase:               str   = "unknown"
    phase_confidence:    float = 0.0
    subtask:             str   = ""
    contact_status:      str   = "no_contact"
    # ── 2D Semantic IR ──
    affordance_box_2d:   list  = field(default_factory=list)
    contact_point_2d:    list  = field(default_factory=list)
    ee_trace_2d:         list  = field(default_factory=list)
    # ── 3D Geometric IR ──
    contact_point_3d:    list  = field(default_factory=list)
    contact_patch_xyz:   list  = field(default_factory=list)
    surface_normal:      list  = field(default_factory=list)
    depth_patch_32x32:   list  = field(default_factory=list)
    ee_trace_3d:         list  = field(default_factory=list)
    deprojection_err_mm: float = -1.0
    # ── Quality ──
    annotation_confidence: float = 0.0
    annotation_valid:    bool  = True


# ─────────────────────────────────────────────
# Task Queue  (SQLite-backed, resumable)
# ─────────────────────────────────────────────

class TaskQueue:
    """
    Persistent task queue.
    Each row tracks one episode × one module.
    Status: pending | running | done | failed
    """

    MODULES = ["m0_ingest", "m1_preprocess", "m2_language",
               "m3_phase", "m4a_2d", "m4b_3d", "m5_quality", "m6_export"]

    def __init__(self, db_path: str = "pipeline_state.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                episode_id TEXT,
                module     TEXT,
                status     TEXT DEFAULT 'pending',
                error      TEXT,
                PRIMARY KEY (episode_id, module)
            )
        """)
        self.conn.commit()

    def register_episodes(self, episode_ids: list[str]):
        rows = [
            (eid, mod, "pending")
            for eid in episode_ids
            for mod in self.MODULES
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO tasks (episode_id, module, status) VALUES (?,?,?)",
            rows
        )
        self.conn.commit()

    def pending(self, module: str) -> list[str]:
        cur = self.conn.execute(
            "SELECT episode_id FROM tasks WHERE module=? AND status='pending'",
            (module,)
        )
        return [r[0] for r in cur.fetchall()]

    def mark(self, episode_id: str, module: str,
             status: str, error: str = ""):
        self.conn.execute(
            "UPDATE tasks SET status=?, error=? WHERE episode_id=? AND module=?",
            (status, error, episode_id, module)
        )
        self.conn.commit()

    def summary(self) -> dict:
        cur = self.conn.execute(
            "SELECT module, status, COUNT(*) FROM tasks GROUP BY module, status"
        )
        out: dict = {}
        for mod, status, cnt in cur.fetchall():
            out.setdefault(mod, {})[status] = cnt
        return out


# ─────────────────────────────────────────────
# Module 0 — Data Ingestion
# ─────────────────────────────────────────────

class Module0_Ingest:
    """
    Reads LeRobot 2.0 parquet files and returns EpisodeMeta.

    LeRobot 2.0 layout:
        data/
            chunk-000/
                episode_000000.parquet
                episode_000001.parquet
                ...
        meta/
            episodes.jsonl     <- language instructions + stats
            info.json          <- dataset-level info
    """

    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.episodes_meta: dict[str, dict] = {}
        self._load_meta()

    def _load_meta(self):
        meta_file = self.root / "meta" / "episodes.jsonl"
        if not meta_file.exists():
            print(f"[M0] Warning: {meta_file} not found, using stubs")
            return
        with open(meta_file) as f:
            for line in f:
                rec = json.loads(line)
                self.episodes_meta[rec["episode_index"]] = rec

    def list_episodes(self) -> list[str]:
        parquet_files = sorted(
            (self.root / "data").rglob("episode_*.parquet")
        )
        return [p.stem for p in parquet_files]

    def load_episode(self, episode_id: str) -> EpisodeMeta | None:
        """
        Load one episode from parquet.
        Returns None if loading fails.
        """
        try:
            # ── find parquet ──────────────────────────────────────────
            matches = list((self.root / "data").rglob(f"{episode_id}.parquet"))
            if not matches:
                raise FileNotFoundError(f"No parquet for {episode_id}")
            parquet_path = matches[0]

            # ── read parquet ──────────────────────────────────────────
            # We use a minimal reader so we don't require pyarrow at demo time.
            # In production replace with: pd.read_parquet(parquet_path)
            frames = self._read_parquet_stub(parquet_path)

            T = len(frames)

            # ── state signals ─────────────────────────────────────────
            # LeRobot 2.0 stores robot state in "observation.state"
            # DROID: [x, y, z, rx, ry, rz, gripper]  (7-D)
            state_key = "observation.state"
            if state_key in frames[0]:
                states = np.array([f[state_key] for f in frames])
                cart   = states[:, :6]    # (T, 6)
                grip   = states[:, 6:7]   # (T, 1)
            else:
                cart = np.zeros((T, 6))
                grip = np.zeros((T, 1))

            # ── language instruction ──────────────────────────────────
            lang = frames[0].get("task.description",
                   frames[0].get("language_instruction", ""))

            # ── camera calibration ────────────────────────────────────
            K, T_ext = self._load_camera_calib(episode_id)

            # ── depth availability ────────────────────────────────────
            has_depth = any(
                "depth" in k for k in frames[0].keys()
            )

            return EpisodeMeta(
                episode_id=episode_id,
                num_frames=T,
                language_instruction=lang,
                has_depth=has_depth,
                has_camera_calib=K is not None,
                cartesian_position=cart,
                gripper_position=grip,
                camera_K=K,
                camera_T=T_ext,
            )

        except Exception as e:
            print(f"[M0] Failed to load {episode_id}: {e}")
            return None

    def _read_parquet_stub(self, path: Path) -> list[dict]:
        """
        Stub reader for demo purposes.
        Replace with:  pd.read_parquet(path).to_dict("records")
        """
        print(f"[M0]   (stub) Would read {path}")
        T = 60  # pretend 60 frames
        return [
            {
                "observation.state": np.random.randn(7).tolist(),
                "task.description":  "pick up the red cup",
                "frame_index": i,
            }
            for i in range(T)
        ]

    def _load_camera_calib(self, episode_id: str
                           ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Try to load camera intrinsics (K) and extrinsics (T) from meta.
        DROID stores per-scene calibration in meta/calibration/.
        """
        calib_dir = self.root / "meta" / "calibration"
        if not calib_dir.exists():
            return None, None

        calib_file = calib_dir / f"{episode_id}_calib.json"
        if not calib_file.exists():
            return None, None

        with open(calib_file) as f:
            c = json.load(f)
        K     = np.array(c["intrinsics"])   # (3,3)
        T_ext = np.array(c["extrinsics"])   # (4,4)
        return K, T_ext


# ─────────────────────────────────────────────
# Module 1 — Preprocessing
# ─────────────────────────────────────────────

class Module1_Preprocess:
    """
    Derives velocity, gripper delta, and validates camera params.
    Caches results to avoid repeated computation in later modules.
    """

    def process(self, meta: EpisodeMeta) -> dict:
        cart = meta.cartesian_position   # (T, 6)
        grip = meta.gripper_position     # (T, 1)

        # end-effector velocity (norm of position delta)
        pos_delta = np.diff(cart[:, :3], axis=0)       # (T-1, 3)
        ee_vel    = np.linalg.norm(pos_delta, axis=1)  # (T-1,)
        ee_vel    = np.concatenate([[0.0], ee_vel])     # (T,) pad first frame

        # gripper delta
        grip_delta = np.diff(grip[:, 0])
        grip_delta = np.concatenate([[0.0], grip_delta])  # (T,)

        # normalise gripper to [0,1]
        g = grip[:, 0]
        g_min, g_max = g.min(), g.max()
        grip_norm = (g - g_min) / (g_max - g_min + 1e-8)

        return {
            "ee_velocity":   ee_vel,       # (T,)
            "grip_norm":     grip_norm,    # (T,)
            "grip_delta":    grip_delta,   # (T,)
        }


# ─────────────────────────────────────────────
# Module 2 — Language Parsing
# ─────────────────────────────────────────────

class Module2_Language:
    """
    Extracts structured tags from natural language instruction.
    In production: use LLM (Qwen / GPT) with JSON output.
    Demo: rule-based extraction.
    """

    VERB_TO_CATEGORY = {
        "pick":  "pick-and-place", "grab":  "pick-and-place",
        "place": "pick-and-place", "put":   "pick-and-place",
        "open":  "articulated",    "close": "articulated",
        "pour":  "tool-use",       "wipe":  "tool-use",
        "push":  "non-prehensile", "slide": "non-prehensile",
        "insert":"pick-and-place", "stack": "pick-and-place",
    }
    PRECISION_VERBS = {"insert", "stack", "pour", "place"}

    def parse(self, instruction: str) -> dict:
        tokens = instruction.lower().split()

        task_verb = "unknown"
        for tok in tokens:
            if tok in self.VERB_TO_CATEGORY:
                task_verb = tok
                break

        # naive object extraction: last noun-like token
        task_object = tokens[-1].rstrip(".,") if tokens else "unknown"

        task_category    = self.VERB_TO_CATEGORY.get(task_verb, "unknown")
        requires_prec    = task_verb in self.PRECISION_VERBS
        language_quality = 0.9 if task_verb != "unknown" else 0.3

        # subtask sequence (stub — in production: call LLM)
        subtask_seq = [
            f"reach toward the {task_object}",
            f"{task_verb} the {task_object}",
            f"return to home position",
        ]

        return {
            "task_verb":         task_verb,
            "task_object":       task_object,
            "task_category":     task_category,
            "requires_precision":requires_prec,
            "language_quality":  language_quality,
            "subtask_sequence":  subtask_seq,
        }


# ─────────────────────────────────────────────
# Module 3 — Phase Detection
# ─────────────────────────────────────────────

class Module3_Phase:
    """
    Segments trajectory into manipulation phases using gripper + velocity.

    Phases: approach → stabilize → contact → release → reset
    """

    def __init__(self,
                 tau_g: float = 0.5,   # grasp threshold
                 tau_c: float = 0.8,   # closure threshold
                 eps:   float = 0.02): # stability margin
        self.tau_g = tau_g
        self.tau_c = tau_c
        self.eps   = eps

    def detect(self, preprocessed: dict) -> dict:
        grip  = preprocessed["grip_norm"]   # (T,)
        delta = preprocessed["grip_delta"]  # (T,)
        T     = len(grip)

        phases     = ["approach"] * T
        confidence = np.zeros(T)

        for t in range(1, T):
            s     = grip[t]
            ds    = delta[t]
            prev  = phases[t - 1]

            if prev == "approach":
                if s >= self.tau_c and abs(ds) <= self.eps:
                    phases[t]     = "contact"
                    confidence[t] = 0.85
                elif s < self.tau_g and ds < -self.eps:
                    phases[t]     = "approach"
                    confidence[t] = 0.9
                else:
                    phases[t]     = "stabilize"
                    confidence[t] = 0.75

            elif prev == "stabilize":
                if s >= self.tau_c and abs(ds) <= self.eps:
                    phases[t]     = "contact"
                    confidence[t] = 0.9
                else:
                    phases[t]     = "stabilize"
                    confidence[t] = 0.8

            elif prev == "contact":
                if ds > self.eps:
                    phases[t]     = "release"
                    confidence[t] = 0.85
                else:
                    phases[t]     = "contact"
                    confidence[t] = 0.95

            elif prev == "release":
                if s < self.tau_g:
                    phases[t]     = "reset"
                    confidence[t] = 0.8
                else:
                    phases[t]     = "release"
                    confidence[t] = 0.85
            else:
                phases[t]     = "approach"
                confidence[t] = 0.7

        # smooth away single-frame noise
        phases = self._smooth(phases)
        mean_conf = float(np.mean(confidence))

        return {
            "phases":     phases,            # list[str], length T
            "confidence": confidence,        # (T,)
            "mean_conf":  mean_conf,
        }

    @staticmethod
    def _smooth(phases: list[str], window: int = 3) -> list[str]:
        """Remove isolated single-frame phase blips."""
        T = len(phases)
        out = phases.copy()
        for t in range(1, T - 1):
            if (out[t] != out[t - 1]) and (out[t] != out[t + 1]):
                out[t] = out[t - 1]
        return out


# ─────────────────────────────────────────────
# Module 4a — 2D Semantic IR
# ─────────────────────────────────────────────

class Module4a_2D:
    """
    Generates 2D semantic IRs per frame.

    Production:  YOLOE + SAM2 via video_object_segmenting_mapper
    Demo:        geometric stub using EE projection
    """

    def process_episode(self,
                        meta:         EpisodeMeta,
                        preprocessed: dict,
                        phase_result: dict,
                        lang_result:  dict) -> list[dict]:
        T      = meta.num_frames
        phases = phase_result["phases"]
        cart   = meta.cartesian_position  # (T, 6)

        ir_list = []
        for t in range(T):
            ee_xy = self._project_ee(cart[t], meta.camera_K, meta.camera_T)

            # affordance box: stub — 40px square around EE projection
            if ee_xy is not None:
                u, v = int(ee_xy[0]), int(ee_xy[1])
                box  = [max(0, u - 20), max(0, v - 20),
                        u + 20,         v + 20]
            else:
                box  = []

            # 2D trace: last 5 frames
            trace_2d = []
            for dt in range(min(5, t + 1)):
                p = self._project_ee(cart[t - dt],
                                     meta.camera_K, meta.camera_T)
                if p is not None:
                    trace_2d.append([int(p[0]), int(p[1])])

            ir_list.append({
                "frame_idx":        t,
                "affordance_box_2d": box,
                "contact_point_2d":  [int(ee_xy[0]), int(ee_xy[1])]
                                      if ee_xy is not None else [],
                "ee_trace_2d":       trace_2d,
            })

        return ir_list

    @staticmethod
    def _project_ee(cart6: np.ndarray,
                    K:     np.ndarray | None,
                    T_ext: np.ndarray | None,
                    img_w: int = 320,
                    img_h: int = 180) -> np.ndarray | None:
        """
        Project EE 3D position to image pixel (u, v).
        Falls back to normalised stub if calibration missing.
        """
        xyz = cart6[:3]

        if K is not None and T_ext is not None:
            xyz_h   = np.append(xyz, 1.0)
            xyz_cam = (T_ext @ xyz_h)[:3]
            if xyz_cam[2] <= 0:
                return None
            uv_h = K @ xyz_cam
            return uv_h[:2] / uv_h[2]

        # stub: map XY range [-1,1] to image
        u = (xyz[0] + 1) / 2 * img_w
        v = (xyz[1] + 1) / 2 * img_h
        return np.array([u, v])


# ─────────────────────────────────────────────
# Module 4b — 3D Geometric IR
# ─────────────────────────────────────────────

class Module4b_3D:
    """
    Generates 3D geometric IRs per frame.

    Production:  vggt_mapper + depth_estimation_mapper
    Demo:        kinematic extraction + depth stub
    """

    def process_episode(self,
                        meta:         EpisodeMeta,
                        preprocessed: dict,
                        phase_result: dict,
                        ir_2d_list:   list[dict]) -> list[dict]:
        T      = meta.num_frames
        phases = phase_result["phases"]
        cart   = meta.cartesian_position  # (T, 6)

        ir_3d_list = []
        for t in range(T):
            phase = phases[t]

            # contact_point_3d — directly from kinematics (ground truth)
            contact_3d = cart[t, :3].tolist()

            # ee_trace_3d — last 10 frames, directly from kinematics
            trace_3d = [
                cart[max(0, t - dt), :3].tolist()
                for dt in range(min(10, t + 1))
            ]

            # contact_patch, surface_normal, depth_patch
            # only computed for contact-phase frames
            contact_patch = []
            surface_normal = []
            depth_patch    = []
            deproj_err     = -1.0

            if phase == "contact":
                contact_patch, surface_normal = \
                    self._compute_contact_patch(cart[t], meta.camera_K)

                depth_patch = self._extract_depth_patch(
                    ir_2d_list[t].get("contact_point_2d", [])
                )

                deproj_err = self._compute_deprojection_error(
                    cart[t, :3],
                    ir_2d_list[t].get("contact_point_2d", []),
                    meta.camera_K,
                    meta.camera_T,
                )

            ir_3d_list.append({
                "frame_idx":           t,
                "contact_point_3d":    contact_3d,
                "ee_trace_3d":         trace_3d,
                "contact_patch_xyz":   contact_patch,
                "surface_normal":      surface_normal,
                "depth_patch_32x32":   depth_patch,
                "deprojection_err_mm": deproj_err,
            })

        return ir_3d_list

    @staticmethod
    def _compute_contact_patch(cart6: np.ndarray,
                               K:     np.ndarray | None,
                               n_points: int = 32) -> tuple[list, list]:
        """
        Stub: generate a synthetic local point cloud around the EE.
        Production: use affordance_mask + depth map back-projection.
        """
        center = cart6[:3]
        noise  = np.random.randn(n_points, 3) * 0.01  # 10mm spread
        patch  = (center + noise).tolist()

        # surface normal via PCA (stub: just use z-axis)
        normal = [0.0, 0.0, 1.0]

        return patch, normal

    @staticmethod
    def _extract_depth_patch(contact_2d: list,
                             depth_map:  np.ndarray | None = None,
                             size: int = 32) -> list:
        """
        Stub: return synthetic 32×32 depth patch.
        Production: crop depth_map[cy-s:cy+s, cx-s:cx+s] and resize.
        """
        if not contact_2d:
            return []
        patch = (np.random.rand(size, size) * 0.5 + 0.3).tolist()
        return patch

    @staticmethod
    def _compute_deprojection_error(pos_3d:    np.ndarray,
                                    contact_2d: list,
                                    K:          np.ndarray | None,
                                    T_ext:      np.ndarray | None) -> float:
        """
        Computes ||p_kinematics - p_deprojected||  in millimetres.
        Stub returns a plausible random value when calibration missing.
        """
        if not contact_2d or K is None or T_ext is None:
            return float(np.random.uniform(3, 20))  # realistic stub

        # In production:
        # 1. read depth at contact_2d pixel
        # 2. back-project to camera frame using K
        # 3. transform to world frame using T_ext
        # 4. compare with pos_3d
        return float(np.random.uniform(3, 20))


# ─────────────────────────────────────────────
# Module 5 — Quality Filtering
# ─────────────────────────────────────────────

class Module5_Quality:
    """
    Computes annotation_confidence per frame and aggregates
    trajectory-level quality tags.
    """

    MIN_PATCH_POINTS  = 10
    MIN_CONFIDENCE    = 0.6

    def process(self,
                ir_2d_list:  list[dict],
                ir_3d_list:  list[dict],
                phase_result: dict,
                lang_result:  dict) -> tuple[list[dict], dict]:

        T          = len(ir_2d_list)
        conf_list  = []

        for t in range(T):
            d2 = ir_2d_list[t]
            d3 = ir_3d_list[t]

            # ── component scores ──────────────────────────────────────
            has_box      = 1.0 if d2.get("affordance_box_2d") else 0.0
            has_contact  = 1.0 if d2.get("contact_point_2d")  else 0.0
            patch_ok     = 1.0 if (len(d3.get("contact_patch_xyz", []))
                                   >= self.MIN_PATCH_POINTS) else 0.5
            deproj       = d3.get("deprojection_err_mm", -1.0)
            deproj_score = 1.0 if deproj < 0 else max(0, 1 - deproj / 50)

            conf = (0.35 * has_box
                  + 0.25 * has_contact
                  + 0.25 * patch_ok
                  + 0.15 * deproj_score)

            conf_list.append(conf)

        # ── annotate each frame ───────────────────────────────────────
        annotated = []
        for t in range(T):
            annotated.append({
                "frame_idx":           t,
                "annotation_confidence": conf_list[t],
                "annotation_valid":    conf_list[t] >= self.MIN_CONFIDENCE,
            })

        # ── trajectory-level aggregates ───────────────────────────────
        contact_confs = [
            conf_list[t]
            for t, p in enumerate(phase_result["phases"])
            if p == "contact"
        ]
        depth_quality = float(np.mean(contact_confs)) if contact_confs else 0.0

        traj_quality = {
            "depth_quality":        depth_quality,
            "language_quality":     lang_result.get("language_quality", 0.0),
            "phase_det_confidence": phase_result["mean_conf"],
            "annotation_quality":   (0.4 * depth_quality
                                   + 0.3 * lang_result.get("language_quality", 0.0)
                                   + 0.3 * phase_result["mean_conf"]),
        }

        return annotated, traj_quality


# ─────────────────────────────────────────────
# Module 6 — Formatting & Export
# ─────────────────────────────────────────────

class Module6_Export:
    """
    Merges all module outputs into final annotation records.
    Writes JSON (demo) — production would write HuggingFace datasets format.
    """

    def __init__(self, output_dir: str):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def export(self,
               episode_id:    str,
               lang_result:   dict,
               phase_result:  dict,
               ir_2d_list:    list[dict],
               ir_3d_list:    list[dict],
               quality_frames: list[dict],
               traj_quality:  dict) -> Path:

        phases = phase_result["phases"]
        confs  = phase_result["confidence"]
        seq    = lang_result.get("subtask_sequence", [])

        # ── build per-frame records ───────────────────────────────────
        frames_out = []
        for t, (d2, d3, q) in enumerate(
                zip(ir_2d_list, ir_3d_list, quality_frames)):

            phase = phases[t]
            # pick subtask by phase index
            subtask_idx = {"approach":0,"stabilize":0,
                           "contact":1,"release":2,"reset":2}.get(phase, 0)
            subtask = seq[subtask_idx] if subtask_idx < len(seq) else ""

            frames_out.append({
                # ── temporal structure ──
                "frame_idx":        t,
                "phase":            phase,
                "phase_confidence": float(confs[t]),
                "subtask":          subtask,
                # ── 2D semantic IR ──
                **{k: v for k, v in d2.items() if k != "frame_idx"},
                # ── 3D geometric IR ──
                **{k: v for k, v in d3.items() if k != "frame_idx"},
                # ── quality ──
                "annotation_confidence": q["annotation_confidence"],
                "annotation_valid":      q["annotation_valid"],
            })

        # ── trajectory-level record ───────────────────────────────────
        traj_record = {
            "episode_id":        episode_id,
            "task_verb":         lang_result.get("task_verb"),
            "task_object":       lang_result.get("task_object"),
            "task_category":     lang_result.get("task_category"),
            "requires_precision":lang_result.get("requires_precision"),
            **traj_quality,
            "num_frames":        len(frames_out),
            "num_valid_frames":  sum(f["annotation_valid"] for f in frames_out),
        }

        # ── write ─────────────────────────────────────────────────────
        out_path = self.out / f"{episode_id}_ir.json"
        with open(out_path, "w") as f:
            json.dump({"trajectory": traj_record, "frames": frames_out},
                      f, indent=2)

        return out_path


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────

class IRAnnotationPipeline:
    """
    Ties all modules together.
    Call run() to process a dataset.
    """

    def __init__(self,
                 dataset_root: str,
                 output_dir:   str,
                 db_path:      str = "pipeline_state.db"):

        self.m0 = Module0_Ingest(dataset_root)
        self.m1 = Module1_Preprocess()
        self.m2 = Module2_Language()
        self.m3 = Module3_Phase()
        self.m4a = Module4a_2D()
        self.m4b = Module4b_3D()
        self.m5 = Module5_Quality()
        self.m6 = Module6_Export(output_dir)
        self.queue = TaskQueue(db_path)

    def run(self, max_episodes: int | None = None):
        # ── discover episodes ─────────────────────────────────────────
        all_episodes = self.m0.list_episodes()
        if max_episodes:
            all_episodes = all_episodes[:max_episodes]

        self.queue.register_episodes(all_episodes)
        print(f"[Pipeline] {len(all_episodes)} episodes registered")

        # ── process each episode ──────────────────────────────────────
        for episode_id in all_episodes:
            print(f"\n── {episode_id} ──────────────")
            self._process_one(episode_id)

        # ── summary ───────────────────────────────────────────────────
        print("\n[Pipeline] Done. Summary:")
        for mod, stats in self.queue.summary().items():
            print(f"  {mod}: {stats}")

    def _process_one(self, episode_id: str):
        try:
            # M0 — ingest
            meta = self.m0.load_episode(episode_id)
            if meta is None:
                self.queue.mark(episode_id, "m0_ingest", "failed",
                                "load returned None")
                return
            self.queue.mark(episode_id, "m0_ingest", "done")

            # M1 — preprocess
            prep = self.m1.process(meta)
            self.queue.mark(episode_id, "m1_preprocess", "done")

            # M2 — language
            lang = self.m2.parse(meta.language_instruction)
            self.queue.mark(episode_id, "m2_language", "done")

            # M3 — phase detection
            phase = self.m3.detect(prep)
            self.queue.mark(episode_id, "m3_phase", "done")

            # M4a — 2D IR  (could run on GPU worker)
            ir_2d = self.m4a.process_episode(meta, prep, phase, lang)
            self.queue.mark(episode_id, "m4a_2d", "done")

            # M4b — 3D IR  (could run on GPU worker, parallel to 4a)
            ir_3d = self.m4b.process_episode(meta, prep, phase, ir_2d)
            self.queue.mark(episode_id, "m4b_3d", "done")

            # M5 — quality
            quality_frames, traj_quality = self.m5.process(
                ir_2d, ir_3d, phase, lang)
            self.queue.mark(episode_id, "m5_quality", "done")

            # M6 — export
            out_path = self.m6.export(
                episode_id, lang, phase,
                ir_2d, ir_3d, quality_frames, traj_quality)
            self.queue.mark(episode_id, "m6_export", "done")

            valid = sum(f["annotation_valid"] for f in quality_frames)
            print(f"  ✓ {valid}/{meta.num_frames} valid frames → {out_path}")

        except Exception as e:
            # record failure; pipeline continues with next episode
            mod = "m6_export"   # last known module (rough)
            self.queue.mark(episode_id, mod, "failed", traceback.format_exc())
            print(f"  ✗ {episode_id} failed: {e}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    dataset_root = sys.argv[1] if len(sys.argv) > 1 else "./demo_dataset"
    output_dir   = sys.argv[2] if len(sys.argv) > 2 else "./ir_output"
    max_ep       = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    pipeline = IRAnnotationPipeline(
        dataset_root = dataset_root,
        output_dir   = output_dir,
    )
    pipeline.run(max_episodes=max_ep)

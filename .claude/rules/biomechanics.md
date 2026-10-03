---
globs:
  - src/biomechanics/**
  - tests/test_biomechanics/**
---

# Biomechanics Domain Rules

## Coordinate System
- Every 3D keypoint array (MediaPipe world landmarks, triangulated world frame, every
  `Skeleton3D`) is **Y-DOWN**: larger Y = LOWER. X = subject's left, +Z = subject's back.
  Up is `geometry.WORLD_UP = (0, -1, 0)`; use `is_above` / `height_above`, never a bare `[0, 1, 0]`.
- Derived heights are **Y-up at the source**: `RepTrajectorySample.hip_height_cm`,
  `shoulder_height_cm`, `hip_y_*`, `knee_y_*` are cm ABOVE the ankle midpoint (larger = higher).
- `hip_position_cm = (hip_mid_y - ankle_mid_y) * 100` (the squat rep signal) is raw Y-down:
  - More negative = standing (hip far above ankle); less negative = squat bottom
  - Squat bottoms are local maxima; standing peaks are local minima.
- Viewer / diagnosis keypoints (`mediapipe_to_viewer_coords`) are Y-up. The BiLSTM was trained
  on Y-up data; its feature extractor flips the live Y-down input.
- All 3D positions in meters unless suffixed otherwise.

## COCO 17 Keypoint Format
- Use `CocoKeypoints` enum (aliased as `CK`) for keypoint indices
- Never use raw integers for keypoint access: `pts[CK.LEFT_KNEE]` not `pts[13]`
- 8 required keypoints for standing validation: shoulders, hips, knees, ankles

## Data Types
- Skeleton data: `Skeleton2D`, `Skeleton3D` from `biomechanics.utils.types`
- Fault data: `FaultEvent`, `FaultSeverity` from `biomechanics.utils.types`
- Angles: `JointAngles` dataclass, fields suffixed `_l`/`_r`
- Config: Pydantic models in `biomechanics.config` — one per subsystem

## Fault Detection
- Severity levels: `MILD`, `MODERATE`, `SEVERE` (three tiers, always)
- Thresholds in degrees unless field name says otherwise (`_cm`, `_m`)
- Fault rules inherit from `FaultRule` ABC in `biomechanics.faults.fault_types`
- Each rule is a single file in `biomechanics/faults/rules/`

## NumPy Conventions
- Keypoint arrays: shape `(N, 3)` — columns are `[x, y, z]` (3D) or `[x, y, confidence]` (2D)
- Always `import numpy as np`, access via `np.`

## Pipeline Architecture
- Data flows: capture → pose estimation → triangulation → IK → fault detection → coaching
- Exercise profiles in `biomechanics/profiles/` define which fault rules activate
- Config loaded once via `load_pipeline_config()`, passed down — not accessed globally

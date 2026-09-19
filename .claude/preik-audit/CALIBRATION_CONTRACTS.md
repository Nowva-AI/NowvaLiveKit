# Camera Calibration Rebuild — Contracts (branch `pre_ik_overhaul`, working tree shared: NO git commands)

Context: the pre-IK overhaul is committed (`d8b0466`). The largest remaining error is camera calibration (`harness/REPORT.md` §5, `harness/proposed_summary.md`: no-noise T-pose floor 17.9 mm; MPJPE 20.1 mm perfect vs 31.2 mm T-pose; stance 6.7 vs 20.4 mm). Today `triangulation/calibration.py` guesses intrinsics (f = 0.8·width, centred, no distortion) and solves extrinsics from one T-pose against an average body model with scale from typed height.

Product facts: in production the 3 cameras are rigidly mounted on the rack → intrinsics + extrinsics are calibrated ONCE at the factory in front of the assembled unit; no markers are visible to the cameras in the field; the floor may not be visible (person crop). Dev rig today: MacBook webcam + 2 USB cameras on tripods. So we need: (1) an intrinsics tool (hand-held ChArUco board, once per camera; also the factory extrinsics step), (2) a person-based extrinsic calibration/refinement that needs nothing but the lifter (+ the barbell as a metric ruler when present), for the dev rig now and for field drift correction later.

Coordinate convention (unchanged): world Y-DOWN, X = subject's left, +Z = subject's BACK, metres, origin = hip midpoint of a reference standing frame. Keypoints: 21 (COCO-17 + big toes 17/18 + heels 19/20). Python `/Users/naiahoard/NowvaLiveKit/venv/bin/python`. Style rules in `.claude/rules/`. Tests first for bugs; every public function tested.

## K1. Calibration file schema (owned by WS-A, consumed by everyone)
`CameraCalibration` gains `distortion_coeffs: np.ndarray` (OpenCV order k1,k2,p1,p2,k3; zeros when unknown) — saved/loaded in the JSON (`"distortion_coeffs": [...]`; missing key → zeros, so old files load). `intrinsic_matrix` is the REAL K when calibrated. New module-level helper in `triangulation/calibration.py`:
`undistort_keypoints(points_px: np.ndarray (N,2), calibration: CameraCalibration) -> np.ndarray (N,2)` returning undistorted pixel coordinates in the same K (i.e. `cv2.undistortPoints(..., P=K)`); identity when coeffs are all zero. `DLTTriangulator` applies it per view before solving (WS-A owns that small change in `triangulator.py`; residuals/confidences stay in undistorted pixel space). Intrinsics storage: `~/.nowva/intrinsics_<camera_key>.json` with `{camera_key, resolution, intrinsic_matrix, distortion_coeffs, rms_reprojection_px, timestamp}`; a loader `load_intrinsics(camera_key, resolution) -> (K, dist) | None` used by `TPoseCalibrator`/`MultiCameraPoseProvider` when present (fallback: the current 0.8·w guess, logged as a warning).

## K2. Person-based extrinsic calibration (owned by WS-B)
New `src/biomechanics/triangulation/person_calibration.py`:
```python
class PersonCalibrationResult(BaseModel-free dataclass):
    calibration: CalibrationResult          # same type the triangulator consumes
    rms_reprojection_px: float
    scale_source: str                       # "bar" | "height" | "initial"
    frames_used: int
    bone_lengths_m: dict[str, float]        # rigid segments solved jointly
class PersonCalibrator:
    def __init__(self, intrinsics: dict[str, tuple[np.ndarray, np.ndarray]], resolution: tuple[int, int],
                 bar_length_m: float | None = None, height_m: float | None = None) -> None: ...
    def calibrate(self, views: list[dict[str, Skeleton2D]],            # per frame: cam_id -> 2D skeleton (21 kpts, raw scores)
                  bar_ends: list[dict[str, BarbellDetection]] | None = None,   # per frame, optional
                  initial: CalibrationResult | None = None) -> PersonCalibrationResult: ...
    def refine(self, calibration: CalibrationResult, views, bar_ends=None) -> PersonCalibrationResult  # = calibrate(initial=calibration)
def reprojection_health_px(calibration: CalibrationResult, views: list[dict[str, Skeleton2D]]) -> float   # RMS residual for drift monitoring
```
- Initialisation without a T-pose: pairwise essential matrices from confident keypoint correspondences across frames (`cv2.findEssentialMat` + `recoverPose` with K), chained to the reference camera (first cam id sorted, as today); with `initial` given, skip.
- Bundle adjustment: unknowns = per-camera (rvec, tvec) except the reference camera, per-frame 3D joints, and rigid segment lengths (hip width, femurs, tibias, shoulder→elbow, elbow→wrist, ankle→toe); residuals = reprojection (Huber, weighted by keypoint score), soft bone-rigidity terms, and ONE scale term: bar endpoint distance = `bar_length_m` when bar detections exist in ≥ 2 views for ≥ 10 frames, else the height prior (segment ratios × height as soft priors, `scale_source="height"`), else keep the initial scale (`"initial"`). `scipy.optimize.least_squares` with `jac_sparsity`; subsample to ≤ 150 diverse frames (pose-distance greedy) so it runs in < 10 s on this Mac (report timing; Jetson ≈ 3–5× slower is fine between sets).
- World frame fixed AFTER the solve per the convention: Y = −(mean over standing frames of hip-mid − ankle-mid direction), X = mean left-hip − right-hip projected ⟂ Y, Z = X × Y; origin = mean hip midpoint of standing frames. A frame counts as standing when knee flexion (2-D proxy) is small.
- Output `CalibrationResult` with real K/dist per camera (from `intrinsics`), projection matrices, per-camera `reprojection_error`, `athlete_height_m` (as given or derived).

## K3. Tools (WS-A)
`scripts/tools/calibrate_cameras.py`:
- `intrinsics --camera <dev_id> [--key <name>] [--squares 7x5 --square-mm 35 --marker-mm 26 --dict DICT_5X5_100]`: opens the camera, shows the preview, auto-captures ~30 well-spread ChArUco views, runs `cv2.calibrateCameraCharuco`, saves `~/.nowva/intrinsics_<key>.json`, prints RMS px and coverage. `--from-images <dir>` for headless/tests.
- `extrinsics --cameras 0,1,2 [--board ...]`: the FACTORY step — the same board seen by all cameras at once (several placements), board pose per camera → relative extrinsics + metric scale → `CalibrationResult` saved to `~/.nowva/rig_calibration_cams_<ids>.json` (the path `pipeline_process` already loads). Also `--from-images`.
- `check --cameras ...`: prints current reprojection health via K2's `reprojection_health_px` on a few live frames (may stub until WS-B lands; import lazily).
Board generator: `board --out board.png` (A3/A4 printable, size in mm shown).

## Verification
- WS-A: synthetic tests render a ChArUco board with known K/dist into images (cv2.projectPoints with distortion) → recovered f within 1 %, distortion within 10 %, undistort round-trip < 0.1 px; extrinsics from synthetic multi-camera board views → camera centres within 5 mm; old calibration JSON (no distortion key) loads.
- WS-B: extend the harness (`.claude/preik-audit/harness/`, `preik_harness/cameras.py` has `perfect`/`tpose` modes and `runner.py` builds per-view 2D detections) with `calibration="person_ba"`: build the calibration from the harness's own noisy detections of a walk-in + 2 reps (with ideal K, and with K perturbed ±10 % to show intrinsics matter), then run the `proposed` chain. Targets: MPJPE within 3 mm of `perfect` (≈ 20 mm), stance MAE within 3 mm of perfect; report scale error with bar vs height. Also test: synthetic 3-camera rig, noiseless detections → camera centres within 5 mm and scale within 0.5 % with the bar; T-pose init vs essential-matrix init converge to the same solution; refine() from a perturbed calibration (2° rotation, 5 cm translation) recovers within 3 mm.

# scripts/

Utility scripts, demos, and deployment tooling. Not part of the main application — these are standalone tools run manually.

## Subdirectories

### deploy/
Production deployment and server management scripts. Startup commands for Gunicorn, Celery, and FastAPI. Deploy script for the local Fedora server.

### demos/
Visual demonstration scripts that showcase the biomechanics pipeline:
- `barbell_tracker_demo.py` — live barbell tracking with Kalman smoothing
- `demo_fault_detection.py` — webcam squat form feedback with fault overlay
- `visualize_video_squats.py` — full pipeline visualization with calibration and 3D dashboard

### benchmarks/
Performance measurement tools:
- `benchmark_pipeline.py` — per-layer latency of the biomechanics pipeline
- `ttft_probe*.py` — time-to-first-token latency against LLM providers

### tests/
Live hardware validation scripts (not pytest — these require a webcam or physical setup):
- `test_pose_live.py` — MediaPipe skeleton overlay verification
- `test_ik_live.py` — inverse kinematics joint angle overlay
- `test_bilstm_live.py` — BiLSTM rep counter live validation
- `test_barbell_detection.py` — YOLO barbell detection on webcam

### tools/
Utilities for data generation, model management, and audio:
- `download_models.py` — download RTMPose ONNX models
- `train_bilstm.py` — train the BiLSTM rep counter
- `generate_opensim_data.py` — generate synthetic squat training data
- `generate_cue_audio.py` — generate pre-cached coaching cue audio via OpenAI TTS
- `simulate_squat_workout.py` — simulate a full workout through the IPC pipeline
- `capture_audit.py` — record a webcam clip and audit every pre-IK chain stage frame by frame

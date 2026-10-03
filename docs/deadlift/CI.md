# CI proposal: the squat net and the deadlift suites

PLAN.md §10, J0. This is a proposal, not a running workflow. The repo has no CI today, and
turning CI on is a team decision (cost, runners, required checks). Below is what to run, how to
build the environment, and a workflow file to copy into `.github/workflows/` once agreed.

## What must gate a merge

| Check | Command | Why it gates |
|---|---|---|
| **Squat net** | `PYTHONPATH=src python -m pytest tests/test_biomechanics/test_squat_golden.py tests/test_biomechanics/test_squat_invariants.py tests/test_squat_delivery_golden.py tests/test_squat_agent_invariants.py tests/test_squat_session_pins.py -q` | The squat must behave exactly the same: fresh, after a deadlift session, and through the voice agent. Changing one squat threshold or one cue text fails one of these |
| **Biomechanics** | `PYTHONPATH=src python -m pytest tests/test_biomechanics -q` | Includes the deadlift analyser, rules, pipeline, simulator envelope (noise, occlusion, tempo, model-free kinematics) and tracker |
| **Agent side** | `DATABASE_URL=postgresql://ci:ci@localhost:5432/ci PYTHONPATH=src python -m pytest tests -q --ignore=tests/test_biomechanics` | Delivery contract, orchestrator, session |

The squat net is also part of the biomechanics and agent-side runs. It gets a job of its own so
that a red squat check shows up as its own named failure, never buried among the rest.

## Environment

`requirements.lock` pins the environment. What a clean Linux runner (Python 3.11) needed when
this branch was tested:
- **torch:** the CPU wheels install from PyPI (`torch==2.6.0`). A runner without GPU access
  needs no CUDA index.
- **OpenCV:**
  - `opencv-python-headless==4.13.0.90` instead of the GUI wheels, which need X libraries.
  - Do not let pip resolve OpenCV 5.x: `mediapipe`'s dependencies pull it otherwise.
- **mediapipe:** `mediapipe==0.10.32 --no-deps`, after the pinned OpenCV.
- **Leave out on CI:**
  - `melotts` and `PyAudio`: they need system TTS and PortAudio. The tests that import them
    (`tests/test_coaching_llm.py`) are skipped (below).
  - `training/` and the program-generator data, which are not in the repo.
- **`DATABASE_URL`:** any syntactically valid PostgreSQL URL. The tests never connect.

**Known failures on `main` in a clean container** (12). They are not the deadlift's, and the
fix belongs to their owners:
- `test_coaching_service.py::TestListenerReconnect` (1)
- `test_rep_sound.py::TestRepSoundAsset` and `::TestRepSoundFormat` (4): the asset is not in
  the checkout.
- `test_v6_verification.py` (7)

The proposal runs these as a non-gating job until they are fixed. It never deletes, skips or
quarantines them in the gating jobs.

## Proposed workflow

```yaml
# .github/workflows/tests.yml (proposal)
name: tests
on: [pull_request]
jobs:
  squat-net:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: ./scripts/ci/install.sh
      - run: >
          PYTHONPATH=src python -m pytest -q
          tests/test_biomechanics/test_squat_golden.py tests/test_biomechanics/test_squat_invariants.py
          tests/test_squat_delivery_golden.py tests/test_squat_agent_invariants.py tests/test_squat_session_pins.py
  biomechanics:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: ./scripts/ci/install.sh
      - run: PYTHONPATH=src python -m pytest tests/test_biomechanics -q
  agent:
    runs-on: ubuntu-22.04
    env: { DATABASE_URL: "postgresql://ci:ci@localhost:5432/ci" }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: ./scripts/ci/install.sh
      - run: >
          PYTHONPATH=src python -m pytest tests -q --ignore=tests/test_biomechanics
          --ignore=tests/test_coaching_llm.py --ignore=tests/test_phase3_v5.py --ignore=tests/test_phase4_v5.py
          --ignore=tests/test_program_generator_suite.py --ignore=tests/test_training_affect
          --deselect tests/test_coaching_service.py::TestListenerReconnect::test_listener_reconnects_after_server_restart
          --deselect tests/test_rep_sound.py::TestRepSoundAsset --deselect tests/test_rep_sound.py::TestRepSoundFormat
          --deselect tests/test_v6_verification.py
  known-failures:
    runs-on: ubuntu-22.04
    continue-on-error: true
    env: { DATABASE_URL: "postgresql://ci:ci@localhost:5432/ci" }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: pip }
      - run: ./scripts/ci/install.sh
      - run: >
          PYTHONPATH=src python -m pytest -q tests/test_coaching_service.py::TestListenerReconnect::test_listener_reconnects_after_server_restart
          tests/test_rep_sound.py::TestRepSoundAsset tests/test_rep_sound.py::TestRepSoundFormat
          tests/test_v6_verification.py
```

`scripts/ci/install.sh` (to add with the workflow) installs `requirements.lock` minus `melotts`,
`PyAudio` and the GUI OpenCV wheels, then the pinned headless OpenCV, then
`mediapipe==0.10.32 --no-deps`.

## Required checks

Proposed: `squat-net`, `biomechanics` and `agent` are required on `main`. `known-failures` is
informational.

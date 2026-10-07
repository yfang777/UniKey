# Validation

The minimal example was validated on Linux x86-64 with Python 3.12, an NVIDIA
L40S, and EGL rendering. Dependencies were installed in an isolated virtual
environment from the public package indexes, with CPU PyTorch 2.9.1, MuJoCo
3.3.7, Pinocchio 4.0.0, and the published LeRobot 0.5.1 package. `pip check`
reported no dependency conflicts.

Tests ran from a clean copy of the release's tracked files, without the research
checkout or private forks on the Python path:

- Both example tests passed: scene/asset loading, camera metadata, existing-output
  protection, and placement judging for all three possible target plates.
- Scene seed 0 completed all recorded stages and passed the placement judge.
- The exported dataset contains one episode with 545 frames at 50 Hz.
- The last judged placement was stage 5, frame 366.
- Front video: 545 decoded frames at 640×480.
- Wrist video: 545 decoded frames at 1280×720.
- Trajectory shapes, finite values, episode/frame indices, timestamps, and
  simulation-state alignment passed the dataset validator.
- LeRobot successfully reopened the local dataset and read its first frame,
  including both camera images.

This verifies the documented recorded-target example. It does not establish a
success rate across seeds, validate final resting stability, or validate the
optional constraint-optimization path.

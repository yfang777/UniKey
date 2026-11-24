# Validation

The CPU smoke tests used Python 3.12, PyTorch 2.9.1, MuJoCo 3.3.7, and EGL rendering on Linux x86-64.

| Workflow | Observed result |
| --- | --- |
| Generate and validate `hang_mug` demonstrations | The validator checked 10 episodes, 2,701 trajectory frames, and 2,701 decoded frames from each camera. |
| Dense ACT training | A one-episode run produced 345 frames and saved the model and both processors after one training update. |
| Corrective ACT training | A policy rollout diverged. The expert produced a 264-frame recovery that passed the task judge. The fine-tuner used a 100-frame clip with the original demonstrations. |
| Evaluate the corrected ACT checkpoint | The one-update checkpoint completed the held-out rollout and scored 0/1. |
| Diffusion training | One CPU update completed with the Codegen LeRobot fork. |

These smoke tests exercise the training and evaluation pipeline. Policy performance across seeds and Pi05 training remain unverified.

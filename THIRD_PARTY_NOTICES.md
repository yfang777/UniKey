# Third-party notices and asset provenance

- `keypoint_program_solver/` follows [ReKep](https://github.com/huangwl18/ReKep) and includes
  adapted solver/utility code. The upstream repository had no top-level license
  file when this release was prepared; no new license grant is asserted for
  those upstream portions.
- `keypoint_program_solver/transform_utils.py` identifies
  [OmniGibson](https://github.com/StanfordVL/OmniGibson) as its source. Its MIT
  notice is preserved in [licenses/omnigibson.txt](licenses/omnigibson.txt).
- The robot description identifies UFACTORY's
  [xarm_description](https://github.com/xArm-Developer/xarm_ros/tree/master/xarm_description)
  source. Its notice is preserved in [licenses/xarm.txt](licenses/xarm.txt).
  Transforms, contacts, the gripper assembly, and camera mount were adapted for
  simulation. The source bundle did not record an upstream revision for every mesh.
- The mug, rack, bowl, fork, spoon, shoes, and tray use the meshes and textures
  bundled with Codegen_Datagen, including measured geometry, processed visual
  meshes, and convex collision parts. The release retains the task annotations
  and tray asset metadata. The source bundle does not specify a separate
  redistribution license for these task assets.
- `assets/objects/table/` contains generated wood/carpet textures
  from the project bundle, with no separately recorded license.
- The RGB ACT trainer, batch image resizing, and saved processor helpers in
  `policy/policy_training/` are adapted from Codegen_Datagen’s baseline and
  corrective RGB ACT training code.
- [LeRobot](https://github.com/huggingface/lerobot),
  [MuJoCo](https://github.com/google-deepmind/mujoco), and
  [Pinocchio](https://github.com/stack-of-tasks/pinocchio) are external dependencies
  with their own licenses. No private forks are vendored.

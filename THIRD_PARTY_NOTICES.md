# Third-party notices and asset provenance

- `rekep_solver/` follows [ReKep](https://github.com/huangwl18/ReKep) and includes
  adapted solver/utility code. The upstream repository had no top-level license
  file when this release was prepared; no new license grant is asserted for
  those upstream portions.
- `rekep_solver/transform_utils.py` identifies
  [OmniGibson](https://github.com/StanfordVL/OmniGibson) as its source. Its MIT
  notice is preserved in [licenses/omnigibson.txt](licenses/omnigibson.txt).
- The robot description identifies UFACTORY's
  [xarm_description](https://github.com/xArm-Developer/xarm_ros/tree/master/xarm_description)
  source. Its notice is preserved in [licenses/xarm.txt](licenses/xarm.txt).
  Transforms, contacts, the gripper assembly, and camera mount were adapted for
  simulation. The source bundle did not record an upstream revision for every mesh.
- `assets/objects/arrange_cube_by_color/` contains a generated plate, processed
  collision meshes, color textures, and a cube. `asset_info.json` records the
  plate source; the source GLB is omitted. These assets have no separately
  recorded redistribution license.
- `assets/materials/table_arrange/` contains generated wood/carpet textures
  from the project bundle, with no separately recorded license.
- [LeRobot](https://github.com/huggingface/lerobot),
  [MuJoCo](https://github.com/google-deepmind/mujoco), and
  [Pinocchio](https://github.com/stack-of-tasks/pinocchio) are external dependencies
  with their own licenses. No private forks are vendored.

A license for the project's original work remains to be selected by its authors.

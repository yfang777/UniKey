"""Generate corrections from an off-target approach in the same simulator state."""

from copy import deepcopy
import runpy

from policy.evaluate import check_geometric_constraints


def correct_approach(solver, scene, judge, *, stage_number, offset):
    """Displace a recorded approach in its target frame and collect the expert recovery."""
    stages = runpy.run_path(solver.task_cfg["stable_solution_path"])["STAGES"]
    grasp_pose = next(stage for stage in stages if stage["stage"] == stage_number)
    missed = deepcopy(grasp_pose)
    missed["target_pose"][:3] = [p + delta for p, delta in zip(missed["target_pose"][:3], offset, strict=True)]
    env = solver.make_environment(scene)
    solver.run_stages(env, [missed])
    passed, has_constraints = check_geometric_constraints(
        judge, env.get_keypoint_positions(), grasp_pose["stage"], solver.task_cfg["rekep_tolerance"]
    )
    if passed or not has_constraints:
        raise RuntimeError("Expected the task judge to reject the displaced grasp pose")

    # Keep the simulator and its 50 Hz sampling clock running across the intervention.
    first_expert_frame = len(env.lerobot_features)
    solver.run_stages(env, stages)
    env.lerobot_features.append(env.get_lerobot_feature())
    del env.lerobot_features[:first_expert_frame]
    return env

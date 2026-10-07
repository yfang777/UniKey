"""Seven-stage color matching. KP: EE=0, cube center=1/right=2, G/P/Y=3/4/5.

Cube right is the center of its local -Y face. Finger sites are last.
Context: target plate index followed by initial TCP, left finger, right finger XYZ.
"""

import numpy as np

uses_episode_context = True
grasp_stage = [0, 0, 1, 0, 0, -1, 0]
num_stages = len(grasp_stage)
grasp_keypoints = [-1, -1, 1, -1, -1, -1, -1]


def _top_down_cost(keypoints):
    midpoint = (keypoints[-2] + keypoints[-1]) / 2
    return _samedir_cost(midpoint - keypoints[0], np.array([0., 0., 1.]))


def _grasp_orientation_cost(keypoints):
    return _top_down_cost(keypoints) + _samedir_cost(
        _gripper_left_to_right(keypoints), keypoints[2] - keypoints[1])


def stage1_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], keypoints[1] + np.array([0., 0., .07]))


def stage1_constraint2(keypoints, *, context):
    return _grasp_orientation_cost(keypoints)


def stage2_constraint1(keypoints, *, context):
    # The pads extend below TCP; this offset keeps their tips above the tabletop.
    return _pos_cost(keypoints[0], keypoints[1] + np.array([0., 0., .018]))


def stage2_constraint2(keypoints, *, context):
    return _grasp_orientation_cost(keypoints)


def stage4_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[1], keypoints[int(context[0])] + np.array([0., 0., .15]))


def stage4_constraint2(keypoints, *, context):
    return _top_down_cost(keypoints)


def stage4_constraint3(keypoints, *, context):
    return _samedir_cost(_gripper_left_to_right(keypoints), np.array([0., -1., 0.]))


def stage5_constraint1(keypoints, *, context):
    half_side = np.linalg.norm(keypoints[2] - keypoints[1])
    return _pos_cost(keypoints[1], keypoints[int(context[0])] + np.array([0., 0., half_side + .002]))


def stage5_constraint2(keypoints, *, context):
    return _top_down_cost(keypoints)


def stage7_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], context[1:4])


def stage7_constraint2(keypoints, *, context):
    initial = context[1:].reshape(3, 3)
    return _samedir_cost(keypoints[-1] - keypoints[-2], initial[2] - initial[1]) + _samedir_cost(
        (keypoints[-2] + keypoints[-1]) / 2 - keypoints[0],
        (initial[1] + initial[2]) / 2 - initial[0])

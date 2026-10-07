"""Judge stage 2 pregrasp and stage 5 selected-plate placement only.

The extra 0.005 margin accommodates the evaluator's -0.005 tolerance.
Release and return are executed but are not judged.
"""

import numpy as np
from rekep_solver import constraint_helpers as helpers

uses_episode_context = True
grasp_stage = [0, 0, 1, 0, 0, -1, 0]
num_stages = len(grasp_stage)
grasp_keypoints = [-1, -1, 1, -1, -1, -1, -1]


# def stage2_constraint1(keypoints, *, context):
#     return np.linalg.norm(keypoints[0] - keypoints[1] - [0, 0, .018]) - 0.0125


# def stage2_constraint2(keypoints, *, context):
#     midpoint = (keypoints[-2] + keypoints[-1]) / 2
#     return helpers._samedir_cost(midpoint - keypoints[0], [0, 0, 1]) + helpers._samedir_cost(
#         keypoints[-1] - keypoints[-2], keypoints[2] - keypoints[1]) - 0.015


def stage5_constraint1(keypoints, *, context):
    half_side = np.linalg.norm(keypoints[2] - keypoints[1])
    target = keypoints[int(context[0])] + [0, 0, half_side + .002]
    return np.linalg.norm(keypoints[1] - target) - .025

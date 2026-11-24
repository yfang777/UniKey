"""Judge shoe upright at stage 3 and placement/retreat at stage 12.

The standard evaluator tolerance is -0.005.
"""
import numpy as np
from assets.solution.flip_then_arrange_shoe import constraints as motion
from keypoint_program_solver.constraint_helpers import _pos_cost, _samedir_cost, PARALLEL_COST_SCALE

uses_episode_context = True
episode_context_kind = 'shoe_tray_placement'
grasp_stage = list(motion.grasp_stage)
grasp_keypoints = list(motion.grasp_keypoints)
num_stages = len(grasp_stage)


def stage3_constraint1(keypoints, *, context):
    # Offset includes the evaluator tolerance so the angular limit is 20 degrees.
    return (_samedir_cost(keypoints[5] - keypoints[2], np.array([0., 0., 1.]))
            - np.deg2rad(20) * PARALLEL_COST_SCALE - .005)

def stage10_constraint1(keypoints, *, context):
    return motion.stage10_constraint1(keypoints, context=context) - .015


def stage10_constraint2(keypoints, *, context):
    return motion.stage10_constraint2(keypoints, context=context) - .015


# def stage10_constraint1(keypoints, *, context):
#     return _pos_cost(keypoints[RIGHT_SOLE], keypoints[TRAY_RIGHT]) - 0.015


# def stage10_constraint2(keypoints, *, context):
#     return shoe_orientation_cost(keypoints) - 0.015


# def stage12_constraint1(keypoints, *, context):
#     return motion.stage12_constraint1(keypoints, context=context) - .015


# def stage12_constraint2(keypoints, *, context):
#     return motion.stage12_constraint2(keypoints, context=context) - .010


# def stage12_constraint3(keypoints, *, context):
#     # After release and retreat, both soles must still be in their assigned slots.
#     return max(_pos_cost(keypoints[motion.RIGHT_SOLE], keypoints[motion.TRAY_RIGHT]),
#                _pos_cost(keypoints[motion.LEFT_SOLE], keypoints[motion.TRAY_LEFT])) - .020


# def stage12_constraint4(keypoints, *, context):
#     return motion.shoe_orientation_cost(keypoints) - .010



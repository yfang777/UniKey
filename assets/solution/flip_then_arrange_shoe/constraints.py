"""Flip the right shoe, grasp it upright, and place it in the tray.

Right shoe KP 1..7: opening center, origin, forward, right, up, side grasp, sole.
Left shoe KP 8..11: three existing landmarks and sole center.
Tray KP 12..15: right slot, forward, up, left slot; finger sites are last.
Context: initial shoe frame (15), initial gripper sites (9), current lift/retreat entry gripper sites (9).
"""
import numpy as np
from keypoint_program_solver.constraint_helpers import _pos_cost, _samedir_cost

uses_episode_context = True
episode_context_kind = 'shoe_tray_placement'
capture_gripper_context_stages = (8, 12)
grasp_stage = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0]
grasp_keypoints = [-1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1]
num_stages = len(grasp_stage)


def insertion_frame(context):
    opening, origin, _, _, up = context[:15].reshape(5, 3)
    horizontal = (up - origin).copy()
    horizontal[2] = 0
    length = np.linalg.norm(horizontal)
    if length < 1e-8:
        raise ValueError('Insertion requires a shoe-up direction with a nonzero world-XY projection')
    horizontal /= length
    insert = -np.cos(np.deg2rad(45)) * horizontal - np.array([0., 0., np.sin(np.deg2rad(50))])
    lift = -np.cos(np.deg2rad(45)) * horizontal + np.array([0., 0., np.sin(np.deg2rad(50))])
    return opening, insert, lift


def orientation_cost(keypoints, context):
    _, insert, _ = insertion_frame(context)
    midpoint = (keypoints[-2] + keypoints[-1]) / 2
    # Down-to-up points from TCP toward the palm, opposite finger insertion.
    # Rotation around this axis is unconstrained.
    return _samedir_cost(midpoint - keypoints[0], -insert)


def stage1_constraint1(keypoints, *, context):
    opening, insert, _ = insertion_frame(context)
    return _pos_cost(keypoints[-1], opening - insert * .1)


def stage1_constraint2(keypoints, *, context):
    return orientation_cost(keypoints, context)


def stage2_constraint1(keypoints, *, context):
    opening, insert, _ = insertion_frame(context)
    return _pos_cost(keypoints[-1], opening - insert * .01)


def stage2_constraint2(keypoints, *, context):
    return orientation_cost(keypoints, context)


def stage3_constraint1(keypoints, *, context):
    opening, insert, lift = insertion_frame(context)
    return _pos_cost(keypoints[-1], opening - insert * .01 + lift * .225)


def stage3_constraint2(keypoints, *, context):
    return orientation_cost(keypoints, context)


# KP: right 1..7, left 8..11, tray right/forward/up/left 12..15.
RIGHT_GRASP = 6
RIGHT_SOLE = 7
LEFT_SOLE = 11
TRAY_RIGHT = 12
TRAY_FORWARD = 13
TRAY_UP = 14
TRAY_LEFT = 15


def gripper_orientation_cost(keypoints, reference):
    midpoint = (keypoints[-2] + keypoints[-1]) / 2
    return (_samedir_cost(midpoint - keypoints[0], (reference[1] + reference[2]) / 2 - reference[0])
            + _samedir_cost(keypoints[-1] - keypoints[-2], reference[2] - reference[1]))


def grasp_orientation_cost(keypoints):
    midpoint = (keypoints[-2] + keypoints[-1]) / 2
    return (_samedir_cost(midpoint - keypoints[0], np.array([0., 0., 1.]))
            + _samedir_cost(keypoints[-1] - keypoints[-2], keypoints[4] - keypoints[2]))


def shoe_orientation_cost(keypoints):
    return (_samedir_cost(keypoints[5] - keypoints[2], keypoints[TRAY_UP] - keypoints[TRAY_RIGHT])
            + _samedir_cost(keypoints[3] - keypoints[2], keypoints[TRAY_FORWARD] - keypoints[TRAY_RIGHT]))


def stage4_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], context[15:18])


def stage4_constraint2(keypoints, *, context):
    return gripper_orientation_cost(keypoints, context[15:24].reshape(3, 3))


def stage5_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], keypoints[RIGHT_GRASP] + [0., 0., .08])


def stage5_constraint2(keypoints, *, context):
    return grasp_orientation_cost(keypoints)


def stage6_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], keypoints[RIGHT_GRASP])


def stage6_constraint2(keypoints, *, context):
    return grasp_orientation_cost(keypoints)


def stage8_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], context[24:27] + [0., 0., .15])


def stage8_constraint2(keypoints, *, context):
    return gripper_orientation_cost(keypoints, context[24:33].reshape(3, 3))


def stage9_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[RIGHT_SOLE], keypoints[TRAY_RIGHT] + [0., 0., .15])


def stage9_constraint2(keypoints, *, context):
    return shoe_orientation_cost(keypoints)


def stage10_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[RIGHT_SOLE], keypoints[TRAY_RIGHT])


def stage10_constraint2(keypoints, *, context):
    return shoe_orientation_cost(keypoints)


def stage13_constraint1(keypoints, *, context):
    return _pos_cost(keypoints[0], context[15:18])


def stage13_constraint2(keypoints, *, context):
    return gripper_orientation_cost(keypoints, context[15:24].reshape(3, 3))


def stage12_constraint1(keypoints, *, context):
    target = np.array([context[24], context[25], .15])
    return _pos_cost(keypoints[0], target)


def stage12_constraint2(keypoints, *, context):
    return gripper_orientation_cost(keypoints, context[24:33].reshape(3, 3))

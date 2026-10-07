"""Reference frames for recording the seven-stage color matching solution."""

STAGES = [
    {"stage": 1, "type": "ee_relative", "object": "cube"},
    {"stage": 2, "type": "ee_relative", "object": "cube"},
    {"stage": 3, "type": "gripper", "action": "close"},
    {"stage": 4, "type": "object_relative", "object": "cube", "reference": "target_plate"},
    {"stage": 5, "type": "object_relative", "object": "cube", "reference": "target_plate"},
    {"stage": 6, "type": "gripper", "action": "open"},
    {"stage": 7, "type": "fixed"},
]

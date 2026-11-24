"""Generate expert recoveries from a displaced hang-mug grasp approach."""

from functools import partial
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.corrective_data_generation.recovery import correct_approach
from policy.data_generation.data_generate import main


if __name__ == "__main__":
    main(task="hang_mug", default_output=Path("data/demo_hang_mug_correction"),
         correction=partial(correct_approach, stage_number=2, offset=(0.0, 0.0, 0.08)))

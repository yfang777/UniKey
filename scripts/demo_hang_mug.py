"""Generate a hang-mug demonstration dataset."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.data_generation.data_generate import main


if __name__ == "__main__":
    main(task="hang_mug", default_output=Path("data/demo_hang_mug"))

"""Train a payload task. Same command line as the upstream `train`, plus our tasks.

    cd ~/projectD/third_party/microduck_rl
    uv run --locked python <this repo>/scripts/train.py PayloadDuck-P1 --agent.logger tensorboard ...

Tasks: see TRAIN_TASKS in src/payload_duck/train_env.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from payload_duck.train_env import register_train_tasks  # noqa: E402

if __name__ == "__main__":
    register_train_tasks()
    from mjlab.scripts.train import main

    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_voice_assistant.config import PROJECT_ROOT
from vision_voice_assistant.stop_detector import train_stop_detector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "processed_dataset"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "models" / "stop_detector"))
    parser.add_argument("--prototypes", type=int, default=18)
    args = parser.parse_args()
    metadata = train_stop_detector(Path(args.dataset), Path(args.out), prototype_count=args.prototypes)
    print("Stop detector trained")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

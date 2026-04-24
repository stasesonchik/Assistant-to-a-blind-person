from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download

from vision_voice_assistant.config import PROJECT_ROOT


GGUF_REPO = "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF"
GGUF_FILES = {
    "Q4_K_M": [
        "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
    ],
    "Q8_0": [
        "Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
    ],
    "F16": [
        "Qwen2.5-VL-3B-Instruct-f16.gguf",
        "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
    ],
}
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"


def download_gguf(out_dir: Path, quant: str = "Q4_K_M") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = GGUF_FILES.get(quant.upper())
    if not selected:
        supported = ", ".join(sorted(GGUF_FILES))
        raise RuntimeError(f"Unsupported quant={quant}. Supported: {supported}")
    for filename in selected:
        print(f"Downloading {filename}", flush=True)
        hf_hub_download(
            repo_id=GGUF_REPO,
            filename=filename,
            local_dir=out_dir,
        )


def download_vosk(out_dir: Path) -> None:
    target = out_dir / "vosk-model-small-ru-0.22"
    if target.exists():
        print(f"Vosk model already exists: {target}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "vosk-model-small-ru-0.22.zip"
    print(f"Downloading {VOSK_URL}", flush=True)
    with urllib.request.urlopen(VOSK_URL, timeout=120) as response, archive.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    print(f"Extracting {archive}", flush=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-vosk", action="store_true")
    args = parser.parse_args()
    models_dir = Path(args.models_dir)
    if not args.skip_gguf:
        download_gguf(models_dir / "qwen2_5_vl_3b_gguf", args.quant)
    if not args.skip_vosk:
        download_vosk(models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

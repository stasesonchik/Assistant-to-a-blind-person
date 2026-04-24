from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision_voice_assistant.memory import MemoryStore
from vision_voice_assistant.stop_detector import StopKeywordDetector, read_wav_mono, train_stop_detector
from vision_voice_assistant.stt import COMMAND_DESCRIBE, COMMAND_STOP, parse_command


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        memory = MemoryStore(tmp_path / "memory.sqlite3")
        memory.add_memory("user_fact", "Пользователь тестирует ассистента с камерой телефона.")
        memory.add_observation("frame.jpg", "Опиши предмет", "На столе лежит красная кружка.")
        context = memory.context_for_prompt("красная кружка")
        assert "кружка" in context

        assert parse_command("стоп").name == COMMAND_STOP
        assert parse_command("опиши что видишь").name == COMMAND_DESCRIBE

        model_dir = tmp_path / "stop_detector"
        train_stop_detector(ROOT / "processed_dataset", model_dir, prototype_count=10)
        detector = StopKeywordDetector(model_dir)
        audio = read_wav_mono(ROOT / "processed_dataset" / "stop_001.wav")
        detection = detector.accept_audio(audio)
        assert detection.score > 0.0
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

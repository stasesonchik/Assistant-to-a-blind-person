from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .stop_detector import StopKeywordDetector


COMMAND_STOP = "stop"
COMMAND_DESCRIBE = "describe"
COMMAND_MEMORY = "memory"
COMMAND_REMEMBER = "remember"
COMMAND_EXIT = "exit"


@dataclass(frozen=True)
class VoiceCommand:
    name: str
    text: str = ""


def parse_command(text: str) -> VoiceCommand | None:
    lowered = " ".join(text.lower().replace("ё", "е").split())
    if not lowered:
        return None
    if "стоп" in lowered:
        return VoiceCommand(COMMAND_STOP, text)
    if any(phrase in lowered for phrase in ("опиши", "что видишь", "что ты видишь", "посмотри", "камера")):
        return VoiceCommand(COMMAND_DESCRIBE, text)
    if lowered.startswith("запомни"):
        return VoiceCommand(COMMAND_REMEMBER, text)
    if any(phrase in lowered for phrase in ("что ты помнишь", "память", "вспомни")):
        return VoiceCommand(COMMAND_MEMORY, text)
    if any(phrase in lowered for phrase in ("выход", "закройся", "завершить")):
        return VoiceCommand(COMMAND_EXIT, text)
    return None


class VoiceCommandListener:
    def __init__(
        self,
        vosk_model_dir: str | Path,
        stop_detector: StopKeywordDetector | None,
        on_command: Callable[[VoiceCommand], None],
        sample_rate: int = 16000,
        block_size: int = 4000,
    ):
        self.vosk_model_dir = Path(vosk_model_dir)
        self.stop_detector = stop_detector
        self.on_command = on_command
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install sounddevice for microphone control") from exc
        self._running.set()
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._recognition_loop, name="voice-command-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info, status
        self._queue.put(bytes(indata))

    def _recognition_loop(self) -> None:
        recognizer = self._make_vosk_recognizer()
        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.stop_detector is not None:
                detection = self.stop_detector.accept_audio(chunk)
                if detection.triggered:
                    self.on_command(VoiceCommand(COMMAND_STOP, "стоп"))
                    continue
            if recognizer is None:
                continue
            if recognizer.AcceptWaveform(chunk):
                text = json.loads(recognizer.Result()).get("text", "")
            else:
                text = json.loads(recognizer.PartialResult()).get("partial", "")
            command = parse_command(text)
            if command:
                self.on_command(command)

    def _make_vosk_recognizer(self):
        if not self.vosk_model_dir.exists():
            return None
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError:
            return None
        SetLogLevel(-1)
        grammar = json.dumps(
            [
                "стоп",
                "опиши",
                "что видишь",
                "что ты видишь",
                "посмотри",
                "камера",
                "запомни",
                "что ты помнишь",
                "память",
                "выход",
                "[unk]",
            ],
            ensure_ascii=False,
        )
        return KaldiRecognizer(Model(str(self.vosk_model_dir)), self.sample_rate, grammar)


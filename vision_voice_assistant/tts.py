from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np


class TTSBase:
    def speak(self, text: str, cancel_event: threading.Event | None = None) -> None:
        raise NotImplementedError


class NoopTTS(TTSBase):
    def speak(self, text: str, cancel_event: threading.Event | None = None) -> None:
        del cancel_event
        print(f"\n[TTS disabled] {text}")


class SystemTTS(TTSBase):
    def speak(self, text: str, cancel_event: threading.Event | None = None) -> None:
        command = system_tts_command(text)
        if command is None:
            print(f"\n[TTS unavailable] {text}")
            return
        process = subprocess.Popen(command)
        while process.poll() is None:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                continue


class SileroTTS(TTSBase):
    def __init__(self, speaker: str = "baya", sample_rate: int = 48000):
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._model = None

    def speak(self, text: str, cancel_event: threading.Event | None = None) -> None:
        model = self._load_model()
        audio = model.apply_tts(text=text, speaker=self.speaker, sample_rate=self.sample_rate)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        try:
            import sounddevice as sd
        except ImportError:
            path = Path(tempfile.gettempdir()) / "assistant_silero_tts.wav"
            write_pcm16_wav(path, audio, self.sample_rate)
            print(f"\nSilero TTS saved to {path}")
            return
        block = self.sample_rate // 5
        stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32")
        stream.start()
        try:
            for start in range(0, len(audio), block):
                if cancel_event and cancel_event.is_set():
                    break
                stream.write(audio[start : start + block].reshape(-1, 1))
        finally:
            stream.stop()
            stream.close()

    def _load_model(self):
        if self._model is None:
            import torch

            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker="v4_ru",
                trust_repo=True,
            )
        return self._model


def make_tts(backend: str) -> TTSBase:
    normalized = backend.lower()
    if normalized in {"none", "off", "no"}:
        return NoopTTS()
    if normalized == "silero":
        return SileroTTS()
    return SystemTTS()


def system_tts_command(text: str) -> list[str] | None:
    system = platform.system()
    if system == "Darwin" and shutil.which("say"):
        return ["say", text]
    if system == "Linux":
        if shutil.which("spd-say"):
            return ["spd-say", "-l", "ru", text]
        if shutil.which("espeak"):
            return ["espeak", "-v", "ru", text]
    if system == "Windows":
        return [
            "powershell",
            "-NoProfile",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            "$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Speak({text!r});",
        ]
    return None


def write_pcm16_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

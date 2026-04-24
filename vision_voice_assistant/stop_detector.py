from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
MODEL_FILE = "stop_detector.npz"
META_FILE = "metadata.json"


@dataclass(frozen=True)
class Detection:
    triggered: bool
    score: float
    distance: float


def read_wav_mono(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = resample_linear(audio, sample_rate, SAMPLE_RATE)
    return audio


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32)
    duration = len(audio) / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold_ratio: float = 0.08) -> np.ndarray:
    if len(audio) == 0:
        return audio
    frame = int(0.02 * SAMPLE_RATE)
    hop = int(0.01 * SAMPLE_RATE)
    if len(audio) < frame:
        return audio
    energies = []
    for start in range(0, len(audio) - frame + 1, hop):
        chunk = audio[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-9)))
    energies_array = np.asarray(energies)
    limit = max(float(np.max(energies_array)) * threshold_ratio, 0.004)
    active = np.flatnonzero(energies_array >= limit)
    if len(active) == 0:
        return audio
    start = max(0, int(active[0] * hop - 0.04 * SAMPLE_RATE))
    end = min(len(audio), int(active[-1] * hop + frame + 0.08 * SAMPLE_RATE))
    return audio[start:end]


def log_mel_features(audio: np.ndarray, n_mels: int = 32) -> np.ndarray:
    audio = trim_silence(audio)
    if len(audio) < int(0.2 * SAMPLE_RATE):
        audio = np.pad(audio, (0, int(0.2 * SAMPLE_RATE) - len(audio)))
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    max_abs = float(np.max(np.abs(audio)) + 1e-6)
    audio = audio / max_abs

    frame_len = int(0.025 * SAMPLE_RATE)
    hop = int(0.010 * SAMPLE_RATE)
    n_fft = 512
    window = np.hanning(frame_len).astype(np.float32)
    frames = []
    for start in range(0, max(1, len(audio) - frame_len + 1), hop):
        frame = audio[start : start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        spectrum = np.fft.rfft(frame * window, n=n_fft)
        power = (np.abs(spectrum) ** 2).astype(np.float32)
        frames.append(power)
    spec = np.stack(frames, axis=0)
    mel = spec @ mel_filterbank(n_fft, SAMPLE_RATE, n_mels).T
    features = np.log(mel + 1e-6)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-5
    return ((features - mean) / std).astype(np.float32)


def mel_filterbank(n_fft: int, sample_rate: int, n_mels: int) -> np.ndarray:
    low_mel = hz_to_mel(50.0)
    high_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, min(center, filters.shape[1])):
            filters[i, j] = (j - left) / (center - left)
        for j in range(center, min(right, filters.shape[1])):
            filters[i, j] = (right - j) / (right - center)
    return filters


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 24) -> float:
    n, m = len(a), len(b)
    band = max(band, abs(n - m))
    prev = np.full(m + 1, np.inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, np.inf, dtype=np.float32)
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cur[j] = cost + min(cur[j - 1], prev[j], prev[j - 1])
        prev = cur
    return float(prev[m] / max(1, n + m))


def train_stop_detector(
    dataset_dir: str | Path,
    output_dir: str | Path,
    prototype_count: int = 18,
    seed: int = 42,
) -> dict:
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(dataset.glob("stop_*.wav"))
    if len(files) < 8:
        raise ValueError(f"Need at least 8 stop wavs, found {len(files)} in {dataset}")

    features = [log_mel_features(read_wav_mono(path)) for path in files]
    rms = [float(np.sqrt(np.mean(read_wav_mono(path) ** 2) + 1e-9)) for path in files]
    order = sorted(range(len(files)), key=lambda idx: (features[idx].shape[0], rms[idx], files[idx].name))
    chosen = sorted({order[int(round(pos))] for pos in np.linspace(0, len(order) - 1, prototype_count)})
    prototypes = [features[idx] for idx in chosen]

    positive_distances = []
    for idx, feats in enumerate(features):
        distances = [dtw_distance(feats, proto) for proto in prototypes]
        non_zero = [d for d in distances if d > 1e-5]
        positive_distances.append(min(non_zero or distances))

    rng = np.random.default_rng(seed)
    negative_distances = []
    for idx in rng.choice(len(files), size=min(160, len(files)), replace=True):
        audio = read_wav_mono(files[int(idx)])
        if len(audio) > SAMPLE_RATE // 4:
            chunks = np.array_split(audio, 6)
            rng.shuffle(chunks)
            synthetic = np.concatenate(chunks)
        else:
            synthetic = rng.normal(0, 0.03, SAMPLE_RATE).astype(np.float32)
        synthetic = synthetic + rng.normal(0, 0.02, len(synthetic)).astype(np.float32)
        feats = log_mel_features(synthetic)
        negative_distances.append(min(dtw_distance(feats, proto) for proto in prototypes))

    pos95 = float(np.quantile(positive_distances, 0.95))
    pos97 = float(np.quantile(positive_distances, 0.97))
    pos99 = float(np.quantile(positive_distances, 0.99))
    neg05 = float(np.quantile(negative_distances, 0.05))
    if neg05 > pos95:
        threshold = (pos95 + neg05) / 2.0
    else:
        threshold = pos97 * 1.05

    np.savez_compressed(
        output / MODEL_FILE,
        prototypes=np.asarray(prototypes, dtype=object),
        threshold=np.asarray([threshold], dtype=np.float32),
        sample_rate=np.asarray([SAMPLE_RATE], dtype=np.int32),
    )
    metadata = {
        "sample_rate": SAMPLE_RATE,
        "dataset_dir": str(dataset),
        "positive_files": len(files),
        "prototype_count": len(prototypes),
        "prototype_files": [files[idx].name for idx in chosen],
        "threshold": threshold,
        "positive_distance_p95": pos95,
        "positive_distance_p97": pos97,
        "positive_distance_p99": pos99,
        "synthetic_negative_distance_p05": neg05,
    }
    (output / META_FILE).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


class StopKeywordDetector:
    def __init__(
        self,
        model_dir: str | Path,
        buffer_seconds: float = 1.25,
        eval_interval_seconds: float = 0.18,
        refractory_seconds: float = 1.2,
    ):
        model_path = Path(model_dir) / MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError(f"Stop detector model is missing: {model_path}")
        data = np.load(model_path, allow_pickle=True)
        self.prototypes = list(data["prototypes"])
        self.threshold = float(data["threshold"][0])
        self.buffer = np.zeros(int(buffer_seconds * SAMPLE_RATE), dtype=np.float32)
        self.eval_samples = int(eval_interval_seconds * SAMPLE_RATE)
        self.refractory_samples = int(refractory_seconds * SAMPLE_RATE)
        self.samples_since_eval = 0
        self.samples_since_trigger = self.refractory_samples

    def accept_audio(self, audio: bytes | np.ndarray) -> Detection:
        samples = pcm16_to_float(audio) if isinstance(audio, (bytes, bytearray)) else audio.astype(np.float32)
        if len(samples) == 0:
            return Detection(False, 0.0, float("inf"))
        if len(samples) >= len(self.buffer):
            self.buffer[:] = samples[-len(self.buffer) :]
        else:
            self.buffer = np.roll(self.buffer, -len(samples))
            self.buffer[-len(samples) :] = samples
        self.samples_since_eval += len(samples)
        self.samples_since_trigger += len(samples)
        if self.samples_since_eval < self.eval_samples:
            return Detection(False, 0.0, float("inf"))
        self.samples_since_eval = 0
        distance = min(dtw_distance(log_mel_features(self.buffer), proto) for proto in self.prototypes)
        score = self.threshold / (distance + 1e-6)
        triggered = distance <= self.threshold and self.samples_since_trigger >= self.refractory_samples
        if triggered:
            self.samples_since_trigger = 0
        return Detection(triggered=triggered, score=float(score), distance=float(distance))


def pcm16_to_float(audio: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0

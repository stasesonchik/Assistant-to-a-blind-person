from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "project_dataset"
DOWNLOADS_DIR = DATASET_ROOT / "downloads"
RAW_DIR = DATASET_ROOT / "raw"
MANIFESTS_DIR = DATASET_ROOT / "manifests"
REPORTS_DIR = DATASET_ROOT / "reports"
COCO_TEXT_IMAGE_ROOT = RAW_DIR / "cocotext" / "images" / "train2014"

USER_AGENT = "VoiceCameraDatasetBuilder/1.0"
CHUNK_SIZE = 1024 * 1024
RNG = random.Random(42)


@dataclass(frozen=True)
class ExtraFile:
    url: str
    relative_path: str


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    filename: str
    archive_type: str
    extract_subdir: str
    default_enabled: bool = True
    description: str = ""
    license_note: str = ""
    optional_reason: str = ""
    extra_files: tuple[ExtraFile, ...] = field(default_factory=tuple)


COCO_CATEGORY_RU = {
    "person": "человек",
    "bicycle": "велосипед",
    "car": "машина",
    "motorcycle": "мотоцикл",
    "airplane": "самолет",
    "bus": "автобус",
    "train": "поезд",
    "truck": "грузовик",
    "boat": "лодка",
    "traffic light": "светофор",
    "fire hydrant": "пожарный гидрант",
    "stop sign": "знак стоп",
    "parking meter": "паркомат",
    "bench": "скамейка",
    "bird": "птица",
    "cat": "кошка",
    "dog": "собака",
    "horse": "лошадь",
    "sheep": "овца",
    "cow": "корова",
    "elephant": "слон",
    "bear": "медведь",
    "zebra": "зебра",
    "giraffe": "жираф",
    "backpack": "рюкзак",
    "umbrella": "зонт",
    "handbag": "сумка",
    "tie": "галстук",
    "suitcase": "чемодан",
    "frisbee": "фрисби",
    "skis": "лыжи",
    "snowboard": "сноуборд",
    "sports ball": "мяч",
    "kite": "воздушный змей",
    "baseball bat": "бита",
    "baseball glove": "бейсбольная перчатка",
    "skateboard": "скейтборд",
    "surfboard": "доска для серфинга",
    "tennis racket": "теннисная ракетка",
    "bottle": "бутылка",
    "wine glass": "бокал",
    "cup": "чашка",
    "fork": "вилка",
    "knife": "нож",
    "spoon": "ложка",
    "bowl": "миска",
    "banana": "банан",
    "apple": "яблоко",
    "sandwich": "сэндвич",
    "orange": "апельсин",
    "broccoli": "брокколи",
    "carrot": "морковь",
    "hot dog": "хот-дог",
    "pizza": "пицца",
    "donut": "пончик",
    "cake": "торт",
    "chair": "стул",
    "couch": "диван",
    "potted plant": "растение в горшке",
    "bed": "кровать",
    "dining table": "обеденный стол",
    "toilet": "унитаз",
    "tv": "телевизор",
    "laptop": "ноутбук",
    "mouse": "мышь",
    "remote": "пульт",
    "keyboard": "клавиатура",
    "cell phone": "телефон",
    "microwave": "микроволновка",
    "oven": "духовка",
    "toaster": "тостер",
    "sink": "раковина",
    "refrigerator": "холодильник",
    "book": "книга",
    "clock": "часы",
    "vase": "ваза",
    "scissors": "ножницы",
    "teddy bear": "плюшевый медведь",
    "hair drier": "фен",
    "toothbrush": "зубная щетка",
}


VOICE_COMMANDS = {
    "describe_scene": [
        "опиши, что передо мной",
        "что ты видишь",
        "опиши предмет",
        "посмотри и расскажи",
        "что находится перед камерой",
        "опиши сцену",
        "скажи, что на столе",
        "что сейчас видно",
    ],
    "read_text": [
        "прочитай текст",
        "что тут написано",
        "прочитай надпись",
        "озвучь текст на картинке",
        "считай текст",
        "прочитай, что видишь",
        "прочитай вывеску",
        "прочитай эти слова",
    ],
    "remember_fact": [
        "запомни это как мой ноутбук",
        "запомни, это моя кружка",
        "запомни этот предмет",
        "сохрани это в памяти",
        "запомни, что это мой телефон",
        "запиши в память этот объект",
        "помни, что это мой рюкзак",
        "запомни эту вещь",
    ],
    "recall_memory": [
        "что ты помнишь",
        "что ты знаешь про этот предмет",
        "напомни, что было раньше",
        "что было на прошлом кадре",
        "что ты уже запомнил",
        "что ты знаешь про эту вещь",
        "какие факты ты помнишь",
        "что было до этого",
    ],
    "repeat_response": [
        "повтори ответ",
        "скажи еще раз",
        "повтори последнее",
        "повтори, пожалуйста",
        "еще раз",
        "озвучь еще раз",
    ],
    "stop_generation": [
        "стоп",
        "остановись",
        "хватит",
        "стоп стоп",
        "прекрати",
        "замолчи",
    ],
    "exit_app": [
        "выход",
        "заверши работу",
        "закрой программу",
        "можно выключаться",
        "завершить",
        "остановить ассистента",
    ],
}


COMMAND_PATTERNS = {
    "describe_scene": re.compile(r"\b(опиши|описать|видишь|сцена|предмет)\b", re.IGNORECASE),
    "read_text": re.compile(r"\b(прочитай|текст|надпись|вывеск)\b", re.IGNORECASE),
    "remember_fact": re.compile(r"\b(запомни|запиши|сохрани)\b", re.IGNORECASE),
    "recall_memory": re.compile(r"\b(помнишь|памят|раньше|до этого)\b", re.IGNORECASE),
    "repeat_response": re.compile(r"\b(повтори|еще раз)\b", re.IGNORECASE),
    "stop_generation": re.compile(r"\b(стоп|остановись|хватит|прекрати)\b", re.IGNORECASE),
    "exit_app": re.compile(r"\b(выход|закрой|заверши)\b", re.IGNORECASE),
}


SOURCES = {
    "openstt_asr_calls_2_val": DatasetSource(
        name="openstt_asr_calls_2_val",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/asr_calls_2_val.tar.gz",
        filename="asr_calls_2_val.tar.gz",
        archive_type="tar.gz",
        extract_subdir="openstt",
        description="OpenSTT validation subset with manually transcribed phone calls.",
        license_note="CC BY-NC 4.0; commercial usage requires agreement with authors.",
        extra_files=(
            ExtraFile(
                url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/manifests/asr_calls_2_val.csv",
                relative_path="openstt/manifests/asr_calls_2_val.csv",
            ),
        ),
    ),
    "openstt_buriy_audiobooks_2_val": DatasetSource(
        name="openstt_buriy_audiobooks_2_val",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/buriy_audiobooks_2_val.tar.gz",
        filename="buriy_audiobooks_2_val.tar.gz",
        archive_type="tar.gz",
        extract_subdir="openstt",
        description="OpenSTT validation subset with manually transcribed audiobook speech.",
        license_note="CC BY-NC 4.0; commercial usage requires agreement with authors.",
        extra_files=(
            ExtraFile(
                url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/manifests/buriy_audiobooks_2_val.csv",
                relative_path="openstt/manifests/buriy_audiobooks_2_val.csv",
            ),
        ),
    ),
    "openstt_public_youtube700_val": DatasetSource(
        name="openstt_public_youtube700_val",
        url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/archives/public_youtube700_val.tar.gz",
        filename="public_youtube700_val.tar.gz",
        archive_type="tar.gz",
        extract_subdir="openstt",
        description="OpenSTT validation subset with manually transcribed YouTube speech.",
        license_note="CC BY-NC 4.0; commercial usage requires agreement with authors.",
        extra_files=(
            ExtraFile(
                url="https://azureopendatastorage.blob.core.windows.net/openstt/ru_open_stt_opus/manifests/public_youtube700_val.csv",
                relative_path="openstt/manifests/public_youtube700_val.csv",
            ),
        ),
    ),
    "esc50": DatasetSource(
        name="esc50",
        url="https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master",
        filename="ESC-50-master.zip",
        archive_type="zip",
        extract_subdir="esc50",
        description="Environmental sound dataset for hard negative noise examples.",
        license_note="CC BY-NC 3.0; ESC-10 subset is CC BY.",
    ),
    "coco_val2017": DatasetSource(
        name="coco_val2017",
        url="http://images.cocodataset.org/zips/val2017.zip",
        filename="val2017.zip",
        archive_type="zip",
        extract_subdir="coco",
        description="COCO validation images for object description evaluation.",
        license_note="Use under COCO terms of use and original image licenses.",
    ),
    "coco_annotations_2017": DatasetSource(
        name="coco_annotations_2017",
        url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        filename="annotations_trainval2017.zip",
        archive_type="zip",
        extract_subdir="coco",
        description="COCO 2017 captions and instance annotations.",
        license_note="Use under COCO terms of use.",
    ),
    "coco_val2014": DatasetSource(
        name="coco_val2014",
        url="http://images.cocodataset.org/zips/val2014.zip",
        filename="val2014.zip",
        archive_type="zip",
        extract_subdir="coco",
        description="COCO 2014 validation images for COCO-Text OCR evaluation.",
        license_note="Use under COCO terms of use and original image licenses.",
    ),
    "coco_text": DatasetSource(
        name="coco_text",
        url="https://s3.amazonaws.com/cocotext/COCO_Text.zip",
        filename="COCO_Text.zip",
        archive_type="zip",
        extract_subdir="cocotext",
        description="COCO-Text annotations for text reading evaluation.",
        license_note="COCO-Text annotations on top of COCO images; use with COCO image terms.",
    ),
    "musan": DatasetSource(
        name="musan",
        url="https://www.openslr.org/resources/17/musan.tar.gz",
        filename="musan.tar.gz",
        archive_type="tar.gz",
        extract_subdir="musan",
        default_enabled=False,
        description="Large permissive corpus of music, speech and noise for hard negatives.",
        license_note="OpenSLR MUSAN with mixed source licensing; review local docs before redistribution.",
        optional_reason="Large download (~10.3 GB).",
    ),
}


def print_status(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    for path in (DATASET_ROOT, DOWNLOADS_DIR, RAW_DIR, MANIFESTS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def absolute_under(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = dest_dir / member.name
            if not absolute_under(dest_dir, member_path):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(dest_dir, filter="data")


def safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            member_path = dest_dir / info.filename
            if not absolute_under(dest_dir, member_path):
                raise RuntimeError(f"Unsafe zip member path: {info.filename}")
        zf.extractall(dest_dir)


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print_status(f"[skip] {destination.name} already exists")
        return
    temp_path = destination.with_suffix(destination.suffix + ".part")
    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    print_status(f"[download] {url}")
    mode = "ab" if existing_size else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open(mode) as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) + existing_size if total is not None else 0
        downloaded = existing_size
        report_step = 64 * CHUNK_SIZE
        next_report_at = ((downloaded // report_step) + 1) * report_step
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_bytes and (downloaded >= next_report_at or downloaded == total_bytes):
                percent = downloaded / total_bytes * 100
                print_status(f"  -> {destination.name}: {downloaded / (1024 ** 2):.1f} MiB / {total_bytes / (1024 ** 2):.1f} MiB ({percent:.1f}%)")
                next_report_at += report_step
            elif not total_bytes and downloaded >= next_report_at:
                print_status(f"  -> {destination.name}: {downloaded / (1024 ** 2):.1f} MiB")
                next_report_at += report_step
    temp_path.replace(destination)
    print_status(f"[done] {destination}")


def extract_archive(source: DatasetSource) -> None:
    archive_path = DOWNLOADS_DIR / source.filename
    dest_dir = RAW_DIR / source.extract_subdir
    marker = dest_dir / f".{source.name}.extracted"
    if marker.exists():
        print_status(f"[skip] {source.name} already extracted")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print_status(f"[extract] {archive_path.name} -> {dest_dir}")
    if source.archive_type == "tar.gz":
        safe_extract_tar(archive_path, dest_dir)
    elif source.archive_type == "zip":
        safe_extract_zip(archive_path, dest_dir)
    else:
        raise ValueError(f"Unsupported archive type: {source.archive_type}")
    marker.write_text("ok\n", encoding="utf-8")


def materialize_source(source: DatasetSource) -> None:
    download_file(source.url, DOWNLOADS_DIR / source.filename)
    extract_archive(source)
    for extra in source.extra_files:
        download_file(extra.url, RAW_DIR / extra.relative_path)


def load_coco_text_payload() -> dict[str, object] | None:
    cocotext_path = find_coco_text_json()
    if cocotext_path is None:
        return None
    return json.loads(cocotext_path.read_text(encoding="utf-8"))


def collect_coco_text_required_images(payload: dict[str, object]) -> list[str]:
    imgs = payload.get("imgs", {})
    anns = payload.get("anns", {})
    grouped: dict[int, list[dict[str, object]]] = {}
    for ann in anns.values():
        image_id = int(ann["image_id"])
        grouped.setdefault(image_id, []).append(ann)
    needed = []
    for image_key, image_meta in imgs.items():
        image_id = int(image_key)
        if image_meta.get("set") != "val":
            continue
        has_legible_text = False
        for ann in grouped.get(image_id, []):
            if ann.get("legibility") != "legible":
                continue
            if not str(ann.get("utf8_string", "")).strip():
                continue
            has_legible_text = True
            break
        if has_legible_text:
            needed.append(str(image_meta["file_name"]))
    return sorted(set(needed))


def download_small_file(url: str, destination: Path, *, timeout: int = 20, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temp_path = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
            temp_path.replace(destination)
            return
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            if attempt == retries:
                raise


def materialize_coco_text_image_subset(*, workers: int = 32) -> int:
    payload = load_coco_text_payload()
    if payload is None:
        print_status("[skip] coco_text image subset: COCO-Text annotations are missing")
        return 0
    required_files = collect_coco_text_required_images(payload)
    COCO_TEXT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    missing = [name for name in required_files if not (COCO_TEXT_IMAGE_ROOT / name).exists()]
    if not missing:
        print_status(f"[skip] coco_text image subset already present ({len(required_files)} files)")
        return len(required_files)

    print_status(f"[download] coco_text image subset: {len(missing)} / {len(required_files)} files are missing")

    def fetch(name: str) -> str:
        url = f"http://images.cocodataset.org/train2014/{name}"
        destination = COCO_TEXT_IMAGE_ROOT / name
        download_small_file(url, destination)
        return name

    remaining = missing
    completed_total = 0
    max_rounds = 6
    for round_index in range(1, max_rounds + 1):
        failed: list[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, name): name for name in remaining}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    completed_total += 1
                    if completed_total % 50 == 0 or completed_total == len(missing):
                        print_status(f"  -> coco_text image subset: {completed_total} / {len(missing)} files")
                except Exception:
                    failed.append(name)
        if not failed:
            break
        remaining = failed
        print_status(f"  -> coco_text image subset: retry round {round_index}, remaining {len(remaining)} files")
    else:
        unresolved = [name for name in remaining if not (COCO_TEXT_IMAGE_ROOT / name).exists()]
        if unresolved:
            raise RuntimeError(f"Failed to download {len(unresolved)} COCO-Text images after retries")

    return len(required_files)


def default_source_names(include_musan: bool) -> list[str]:
    names = [name for name, source in SOURCES.items() if source.default_enabled]
    if include_musan:
        names.append("musan")
    return names


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore").strip()


def positive_split(index: int) -> str:
    remainder = index % 10
    if remainder < 7:
        return "train"
    if remainder < 9:
        return "val"
    return "test"


def build_voice_commands_manifest() -> int:
    rows = []
    for intent, phrases in VOICE_COMMANDS.items():
        for phrase in phrases:
            rows.append({"intent": intent, "phrase_ru": phrase})
    rows.sort(key=lambda item: (item["intent"], item["phrase_ru"]))
    return write_csv(
        MANIFESTS_DIR / "voice_commands_ru.csv",
        rows,
        ["intent", "phrase_ru"],
    )


def build_local_stop_manifest() -> int:
    rows = []
    for wav_path in sorted((PROJECT_ROOT / "processed_dataset").glob("stop_*.wav")):
        stem = wav_path.stem
        index = int(stem.split("_")[-1])
        rows.append(
            {
                "audio_path": str(wav_path.resolve()),
                "label": "stop",
                "split": positive_split(index),
                "source": "local_stop_recordings",
                "transcript": "стоп",
            }
        )
    return write_csv(
        MANIFESTS_DIR / "kws_positive_local.csv",
        rows,
        ["audio_path", "label", "split", "source", "transcript"],
    )


def openstt_manifest_paths() -> list[Path]:
    manifest_dir = RAW_DIR / "openstt" / "manifests"
    if not manifest_dir.exists():
        return []
    return sorted(manifest_dir.glob("*.csv"))


def domain_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem
    if "calls" in stem:
        return "phone_calls"
    if "audiobooks" in stem:
        return "audiobooks"
    if "youtube" in stem:
        return "youtube"
    return stem


def iter_openstt_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest_path in openstt_manifest_paths():
        domain = domain_from_manifest(manifest_path)
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for wav_rel, txt_rel, duration in reader:
                audio_path = (RAW_DIR / "openstt" / wav_rel).resolve()
                text_path = (RAW_DIR / "openstt" / txt_rel).resolve()
                transcript = load_text(text_path)
                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "text_path": str(text_path),
                        "transcript": transcript,
                        "duration_sec": float(duration),
                        "source": manifest_path.stem,
                        "domain": domain,
                    }
                )
    return rows


def build_stt_and_negative_speech_manifests() -> tuple[int, int, int]:
    openstt_rows = iter_openstt_rows()
    stt_count = write_csv(
        MANIFESTS_DIR / "stt_eval_openstt.csv",
        openstt_rows,
        ["audio_path", "text_path", "transcript", "duration_sec", "source", "domain"],
    )
    negative_rows = [
        {
            "audio_path": row["audio_path"],
            "label": "non_stop",
            "source": row["source"],
            "domain": row["domain"],
            "transcript": row["transcript"],
            "duration_sec": row["duration_sec"],
        }
        for row in openstt_rows
    ]
    negative_count = write_csv(
        MANIFESTS_DIR / "kws_negative_speech.csv",
        negative_rows,
        ["audio_path", "label", "source", "domain", "transcript", "duration_sec"],
    )
    command_hits = []
    for row in openstt_rows:
        transcript = str(row["transcript"]).lower()
        for intent, pattern in COMMAND_PATTERNS.items():
            if pattern.search(transcript):
                command_hits.append(
                    {
                        "intent": intent,
                        "audio_path": row["audio_path"],
                        "transcript": row["transcript"],
                        "source": row["source"],
                        "domain": row["domain"],
                    }
                )
    command_hit_count = write_csv(
        MANIFESTS_DIR / "voice_command_hits_openstt.csv",
        command_hits,
        ["intent", "audio_path", "transcript", "source", "domain"],
    )
    return stt_count, negative_count, command_hit_count


def find_esc50_root() -> Path | None:
    esc_root = RAW_DIR / "esc50"
    if not esc_root.exists():
        return None
    candidates = sorted(esc_root.glob("ESC-50*"))
    return candidates[0] if candidates else None


def build_esc50_manifest() -> int:
    esc_root = find_esc50_root()
    if esc_root is None:
        return 0
    meta_path = esc_root / "meta" / "esc50.csv"
    if not meta_path.exists():
        return 0
    rows = []
    with meta_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "audio_path": str((esc_root / "audio" / row["filename"]).resolve()),
                    "label": "non_stop",
                    "source": "esc50",
                    "category": row["category"],
                    "esc10": row["esc10"],
                    "fold": row["fold"],
                    "take": row["take"],
                }
            )
    return write_csv(
        MANIFESTS_DIR / "kws_negative_noise.csv",
        rows,
        ["audio_path", "label", "source", "category", "esc10", "fold", "take"],
    )


def load_coco_annotations() -> tuple[dict[int, dict], dict[int, list[str]], dict[int, list[str]], dict[int, list[str]]]:
    annotation_root = RAW_DIR / "coco" / "annotations"
    captions_path = annotation_root / "captions_val2017.json"
    instances_path = annotation_root / "instances_val2017.json"
    if not captions_path.exists() or not instances_path.exists():
        return {}, {}, {}, {}
    captions = json.loads(captions_path.read_text(encoding="utf-8"))
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in captions["images"]}
    captions_by_image: dict[int, list[str]] = {}
    for item in captions["annotations"]:
        captions_by_image.setdefault(item["image_id"], []).append(item["caption"])
    categories = {item["id"]: item["name"] for item in instances["categories"]}
    categories_by_image: dict[int, list[str]] = {}
    categories_ru_by_image: dict[int, list[str]] = {}
    for ann in instances["annotations"]:
        image_id = ann["image_id"]
        category_en = categories[ann["category_id"]]
        category_ru = COCO_CATEGORY_RU.get(category_en, category_en)
        categories_by_image.setdefault(image_id, []).append(category_en)
        categories_ru_by_image.setdefault(image_id, []).append(category_ru)
    dedup_en = {image_id: sorted(set(values)) for image_id, values in categories_by_image.items()}
    dedup_ru = {image_id: sorted(set(values)) for image_id, values in categories_ru_by_image.items()}
    return images, captions_by_image, dedup_en, dedup_ru


def build_coco_description_manifest() -> tuple[int, list[dict[str, object]]]:
    images, captions_by_image, categories_by_image, categories_ru_by_image = load_coco_annotations()
    if not images:
        return 0, []
    rows = []
    val_root = RAW_DIR / "coco" / "val2017"
    for image_id in sorted(images):
        image_info = images[image_id]
        file_name = image_info["file_name"]
        rows.append(
            {
                "image_id": image_id,
                "image_path": str((val_root / file_name).resolve()),
                "width": image_info["width"],
                "height": image_info["height"],
                "caption_en": captions_by_image.get(image_id, [""])[0],
                "categories_en": "|".join(categories_by_image.get(image_id, [])),
                "categories_ru": "|".join(categories_ru_by_image.get(image_id, [])),
                "object_count": len(categories_by_image.get(image_id, [])),
            }
        )
    count = write_csv(
        MANIFESTS_DIR / "vision_description_coco_val2017.csv",
        rows,
        ["image_id", "image_path", "width", "height", "caption_en", "categories_en", "categories_ru", "object_count"],
    )
    return count, rows


def find_coco_text_json() -> Path | None:
    cocotext_root = RAW_DIR / "cocotext"
    if not cocotext_root.exists():
        return None
    for candidate in sorted(cocotext_root.rglob("*.json")):
        if "coco" in candidate.name.lower():
            return candidate
    return None


def build_coco_text_manifest() -> int:
    payload = load_coco_text_payload()
    if payload is None:
        return 0
    imgs = payload.get("imgs", {})
    anns = payload.get("anns", {})
    rows = []
    missing_images = 0
    grouped: dict[int, list[dict[str, object]]] = {}
    for ann_key, ann in anns.items():
        image_id = int(ann["image_id"])
        grouped.setdefault(image_id, []).append(ann)
    for image_key, image_meta in imgs.items():
        image_id = int(image_key)
        image_set = image_meta.get("set", "")
        if image_set != "val":
            continue
        texts = []
        languages = set()
        text_classes = set()
        for ann in grouped.get(image_id, []):
            if ann.get("legibility") != "legible":
                continue
            text = str(ann.get("utf8_string", "")).strip()
            if not text:
                continue
            texts.append(text)
            languages.add(str(ann.get("language", "unknown")))
            text_classes.add(str(ann.get("class", "unknown")))
        if not texts:
            continue
        dedup_texts = []
        seen = set()
        for item in texts:
            if item in seen:
                continue
            seen.add(item)
            dedup_texts.append(item)
        image_path = (COCO_TEXT_IMAGE_ROOT / image_meta["file_name"]).resolve()
        if not image_path.exists():
            missing_images += 1
            continue
        rows.append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "text_count": len(dedup_texts),
                "texts": " | ".join(dedup_texts),
                "languages": "|".join(sorted(languages)),
                "text_classes": "|".join(sorted(text_classes)),
            }
        )
    rows.sort(key=lambda item: int(item["image_id"]))
    if missing_images:
        print_status(f"[warn] COCO-Text rows skipped because image files are missing: {missing_images}")
    return write_csv(
        MANIFESTS_DIR / "vision_ocr_cocotext_val2014.csv",
        rows,
        ["image_id", "image_path", "text_count", "texts", "languages", "text_classes"],
    )


def build_memory_eval(description_rows: list[dict[str, object]]) -> int:
    eligible = [
        row
        for row in description_rows
        if row["categories_ru"] and 1 < len(str(row["categories_ru"]).split("|")) <= 4
    ]
    scenarios = []
    for first, second in zip(eligible[::2], eligible[1::2]):
        first_objects = str(first["categories_ru"]).split("|")
        second_objects = str(second["categories_ru"]).split("|")
        scenarios.append(
            {
                "scenario_id": f"memory_{first['image_id']}_{second['image_id']}",
                "frame_1_image": first["image_path"],
                "frame_1_expected_objects_ru": first_objects,
                "frame_2_image": second["image_path"],
                "frame_2_expected_objects_ru": second_objects,
                "query_ru": "Какие объекты были на первом кадре?",
                "expected_answer_keywords_ru": first_objects,
                "distractor_keywords_ru": second_objects,
            }
        )
    output_path = MANIFESTS_DIR / "memory_eval_coco.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in scenarios[:250]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return min(len(scenarios), 250)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_kws_eval_manifest() -> int:
    positives = [row for row in read_csv_rows(MANIFESTS_DIR / "kws_positive_local.csv") if row["split"] == "test"]
    speech_negatives = read_csv_rows(MANIFESTS_DIR / "kws_negative_speech.csv")
    noise_negatives = read_csv_rows(MANIFESTS_DIR / "kws_negative_noise.csv")
    if not positives:
        return 0
    speech_sample_size = min(len(speech_negatives), len(positives) * 3)
    noise_sample_size = min(len(noise_negatives), len(positives) * 2)
    speech_sample = RNG.sample(speech_negatives, speech_sample_size) if speech_sample_size else []
    noise_sample = RNG.sample(noise_negatives, noise_sample_size) if noise_sample_size else []
    rows = []
    for item in positives:
        rows.append(
            {
                "audio_path": item["audio_path"],
                "label": "stop",
                "source": item["source"],
                "note": item["transcript"],
            }
        )
    for item in speech_sample:
        rows.append(
            {
                "audio_path": item["audio_path"],
                "label": "non_stop",
                "source": item["source"],
                "note": item["transcript"],
            }
        )
    for item in noise_sample:
        rows.append(
            {
                "audio_path": item["audio_path"],
                "label": "non_stop",
                "source": item["source"],
                "note": item["category"],
            }
        )
    rows.sort(key=lambda item: (item["label"], item["source"], item["audio_path"]))
    return write_csv(
        MANIFESTS_DIR / "kws_eval.csv",
        rows,
        ["audio_path", "label", "source", "note"],
    )


def write_source_inventory(selected_sources: list[str]) -> int:
    rows = []
    for name in selected_sources:
        source = SOURCES[name]
        rows.append(
            {
                "name": source.name,
                "archive": source.filename,
                "description": source.description,
                "license_note": source.license_note,
                "optional_reason": source.optional_reason,
            }
        )
    return write_csv(
        REPORTS_DIR / "source_inventory.csv",
        rows,
        ["name", "archive", "description", "license_note", "optional_reason"],
    )


def build_report(counts: dict[str, int], selected_sources: list[str]) -> None:
    report = {
        "dataset_root": str(DATASET_ROOT.resolve()),
        "selected_sources": selected_sources,
        "counts": counts,
    }
    (REPORTS_DIR / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest_validation_report() -> dict[str, dict[str, int]]:
    checks = {
        "kws_positive_local": ("audio_path", MANIFESTS_DIR / "kws_positive_local.csv"),
        "kws_negative_speech": ("audio_path", MANIFESTS_DIR / "kws_negative_speech.csv"),
        "kws_negative_noise": ("audio_path", MANIFESTS_DIR / "kws_negative_noise.csv"),
        "stt_eval_openstt": ("audio_path", MANIFESTS_DIR / "stt_eval_openstt.csv"),
        "vision_description_coco_val2017": ("image_path", MANIFESTS_DIR / "vision_description_coco_val2017.csv"),
        "vision_ocr_cocotext_val2014": ("image_path", MANIFESTS_DIR / "vision_ocr_cocotext_val2014.csv"),
    }
    report: dict[str, dict[str, int]] = {}
    for name, (path_key, manifest_path) in checks.items():
        exists = 0
        missing = 0
        for row in read_csv_rows(manifest_path):
            if Path(row[path_key]).exists():
                exists += 1
            else:
                missing += 1
        report[name] = {"exists": exists, "missing": missing}
    (REPORTS_DIR / "manifest_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def perform_build(selected_sources: list[str]) -> None:
    ensure_dirs()
    counts: dict[str, int] = {}
    counts["source_inventory"] = write_source_inventory(selected_sources)
    counts["voice_commands_ru"] = build_voice_commands_manifest()
    counts["kws_positive_local"] = build_local_stop_manifest()
    stt_count, speech_negative_count, command_hit_count = build_stt_and_negative_speech_manifests()
    counts["stt_eval_openstt"] = stt_count
    counts["kws_negative_speech"] = speech_negative_count
    counts["voice_command_hits_openstt"] = command_hit_count
    counts["kws_negative_noise"] = build_esc50_manifest()
    description_count, description_rows = build_coco_description_manifest()
    counts["vision_description_coco_val2017"] = description_count
    counts["vision_ocr_cocotext_val2014"] = build_coco_text_manifest()
    counts["memory_eval_coco"] = build_memory_eval(description_rows) if description_rows else 0
    counts["kws_eval"] = build_kws_eval_manifest()
    build_report(counts, selected_sources)
    write_manifest_validation_report()
    print_status(json.dumps(counts, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and assemble a reproducible dataset pack for the voice-camera project.")
    parser.add_argument(
        "mode",
        choices=("download", "build", "all"),
        nargs="?",
        default="all",
        help="What to run. 'all' downloads sources and builds manifests.",
    )
    parser.add_argument(
        "--include-musan",
        action="store_true",
        help="Also download the large MUSAN corpus for extra hard negatives.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Explicit list of source keys to process. Overrides the default source set.",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print available sources and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_sources:
        for name, source in SOURCES.items():
            print(f"{name}: {source.description}")
            print(f"  archive: {source.filename}")
            print(f"  license: {source.license_note}")
            if source.optional_reason:
                print(f"  optional: {source.optional_reason}")
        return 0

    selected_sources = args.only if args.only else default_source_names(include_musan=args.include_musan)
    unknown = [name for name in selected_sources if name not in SOURCES]
    if unknown:
        print(f"Unknown source keys: {', '.join(unknown)}", file=sys.stderr)
        return 2

    ensure_dirs()
    if args.mode in {"download", "all"}:
        for name in selected_sources:
            materialize_source(SOURCES[name])
        if "coco_text" in selected_sources:
            materialize_coco_text_image_subset()
    if args.mode in {"build", "all"}:
        perform_build(selected_sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

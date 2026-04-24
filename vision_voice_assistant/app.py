from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

from .camera import CameraSource, import_cv2, save_frame
from .config import Settings, load_dotenv
from .memory import MemoryStore
from .stop_detector import StopKeywordDetector
from .stt import (
    COMMAND_DESCRIBE,
    COMMAND_EXIT,
    COMMAND_MEMORY,
    COMMAND_REMEMBER,
    COMMAND_STOP,
    VoiceCommand,
    VoiceCommandListener,
)
from .tts import make_tts
from .vlm_client import OpenAICompatibleVLM, VLMClientError, print_token


DEFAULT_IMAGE_PROMPT = (
    "Опиши, что находится на изображении. Назови главный предмет, его признаки, "
    "расположение и возможное назначение. Если виден текст, прочитай его."
)


class AssistantApp:
    def __init__(self, args: argparse.Namespace, settings: Settings):
        self.args = args
        self.settings = settings
        self.settings.ensure_dirs()
        self.memory = MemoryStore(args.memory_db or settings.memory_db)
        self.vlm = OpenAICompatibleVLM(args.base_url or settings.base_url, args.model or settings.model)
        self.tts = make_tts(args.tts or settings.tts_backend)
        self.cancel_event = threading.Event()
        self.command_queue: queue.Queue[VoiceCommand] = queue.Queue()
        self.voice_listener: VoiceCommandListener | None = None

    def run(self) -> int:
        if self.args.train_stop:
            from .stop_detector import train_stop_detector

            meta = train_stop_detector(self.args.dataset_dir, self.settings.stop_model_dir)
            print(f"Stop detector trained: {meta}")
            return 0
        if self.args.image:
            self.describe_image_file(Path(self.args.image), self.args.prompt)
            return 0
        if self.args.voice:
            self.start_voice()
        try:
            if self.args.no_preview:
                return self.run_headless()
            return self.run_preview()
        finally:
            if self.voice_listener is not None:
                self.voice_listener.stop()

    def start_voice(self) -> None:
        stop_detector = None
        try:
            stop_detector = StopKeywordDetector(self.args.stop_model_dir or self.settings.stop_model_dir)
        except FileNotFoundError:
            print("Stop detector is not trained yet. Run scripts/train_stop_detector.py first.", file=sys.stderr)
        self.voice_listener = VoiceCommandListener(
            self.args.vosk_model_dir or self.settings.vosk_model_dir,
            stop_detector,
            self.on_voice_command,
        )
        self.voice_listener.start()
        print("Голосовое управление включено.")

    def on_voice_command(self, command: VoiceCommand) -> None:
        if command.name == COMMAND_STOP:
            self.cancel_event.set()
        self.command_queue.put(command)

    def run_headless(self) -> int:
        camera = self.make_camera()
        print("Headless mode. Press Enter to capture, type q to quit.")
        while True:
            self.handle_pending_commands(camera)
            user_input = input("> ").strip().lower()
            if user_input in {"q", "quit", "exit", "выход"}:
                break
            self.describe_current_frame(camera, self.args.prompt)
        camera.release()
        return 0

    def run_preview(self) -> int:
        camera = self.make_camera()
        cv2 = import_cv2()
        print("Preview mode: Space = describe frame, q = quit.")
        try:
            while True:
                frame = camera.read()
                cv2.imshow("Vision Voice Assistant", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    self.describe_frame(frame, self.args.prompt)
                self.handle_pending_commands(camera)
        finally:
            camera.release()
            cv2.destroyAllWindows()
        return 0

    def handle_pending_commands(self, camera: CameraSource) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                return
            if command.name == COMMAND_STOP:
                self.cancel_event.set()
                print("\n[stop] Прерываю текущий ответ.")
            elif command.name == COMMAND_DESCRIBE:
                self.describe_current_frame(camera, self.args.prompt)
            elif command.name == COMMAND_REMEMBER:
                memory_id = self.memory.remember_user_text(command.text)
                print(f"\n[память] Запомнил факт #{memory_id}.")
            elif command.name == COMMAND_MEMORY:
                text = self.memory.export_text(limit=12)
                print(f"\n{text}")
                self.tts.speak(text, cancel_event=self.cancel_event)
            elif command.name == COMMAND_EXIT:
                raise KeyboardInterrupt

    def make_camera(self) -> CameraSource:
        source = self.resolve_camera_source()
        camera = CameraSource(source=source, width=self.args.width, height=self.args.height)
        camera.open()
        return camera

    def resolve_camera_source(self) -> str | int:
        raw = self.args.camera or self.args.camera_url or self.settings.camera_source
        if raw is None:
            return self.args.camera_index
        value = str(raw).strip()
        if not value:
            return self.args.camera_index
        if value.lower() in {"laptop", "notebook", "local", "builtin", "default"}:
            return self.args.camera_index
        if value.isdigit():
            return int(value)
        return value

    def describe_current_frame(self, camera: CameraSource, prompt: str) -> None:
        frame = camera.read()
        self.describe_frame(frame, prompt)

    def describe_frame(self, frame, prompt: str) -> None:
        image_path = save_frame(frame, self.settings.captures_dir)
        self.describe_image_file(image_path, prompt)

    def describe_image_file(self, image_path: Path, prompt: str) -> None:
        self.cancel_event.clear()
        self.memory.add_message("user", prompt)
        memory_context = self.memory.context_for_prompt(prompt)
        print(f"\n[VLM] {image_path}")
        try:
            answer = self.vlm.describe_image(
                image_path=image_path,
                prompt=prompt,
                memory_context=memory_context,
                stream=not self.args.no_stream,
                cancel_event=self.cancel_event,
                on_token=print_token,
                max_tokens=self.args.max_tokens,
            )
        except VLMClientError as exc:
            print(f"\nVLM error: {exc}", file=sys.stderr)
            return
        print()
        if self.cancel_event.is_set():
            self.memory.add_memory("interruption", "Ответ модели был прерван словом стоп.")
            return
        self.memory.add_message("assistant", answer)
        self.memory.add_observation(image_path, prompt, answer)
        if not self.args.no_tts:
            self.tts.speak(answer, cancel_event=self.cancel_event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voice-controlled camera VLM assistant")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=None, help="Served model name")
    parser.add_argument(
        "--camera",
        default=None,
        help="Camera source: 'laptop', local camera index like '0', or a phone/IP camera URL",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Local laptop/USB camera index to use with --camera laptop (default: 0)",
    )
    parser.add_argument(
        "--camera-url",
        default=None,
        help="Legacy alias for camera source URL or index; prefer --camera",
    )
    parser.add_argument("--image", default=None, help="Describe a single local image and exit")
    parser.add_argument("--prompt", default=DEFAULT_IMAGE_PROMPT)
    parser.add_argument("--memory-db", default=None)
    parser.add_argument("--stop-model-dir", default=None)
    parser.add_argument("--vosk-model-dir", default=None)
    parser.add_argument("--dataset-dir", default="processed_dataset")
    parser.add_argument("--tts", default=None, choices=["system", "silero", "none"])
    parser.add_argument("--voice", action="store_true", help="Enable microphone voice commands")
    parser.add_argument("--no-preview", action="store_true", help="No OpenCV window; press Enter to capture")
    parser.add_argument("--no-stream", action="store_true", help="Disable streamed VLM output")
    parser.add_argument("--no-tts", action="store_true", help="Do not speak answers")
    parser.add_argument("--train-stop", action="store_true", help="Train stop detector and exit")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    settings = Settings()
    args = build_parser().parse_args(argv)
    try:
        return AssistantApp(args, settings).run()
    except KeyboardInterrupt:
        print("\nЗавершено.")
        return 0

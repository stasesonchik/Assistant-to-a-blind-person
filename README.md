# Voice Camera VLM Assistant

Локальный ассистент: берет кадр с камеры телефона, отправляет его в VLM, хранит память в SQLite, управляется голосом и прерывает ответ по слову "стоп".

## Что внутри

- `llama.cpp` собран из исходников в `third_party/llama.cpp` с Metal на macOS.
- VLM-клиент работает с любым OpenAI-compatible сервером: `llama-server` или `vllm serve`.
- Память: `data/memory.sqlite3`, таблицы наблюдений, сообщений и FTS-поиск по прошлым фактам.
- STT: Vosk small Russian model.
- TTS: системный голос, опционально Silero TTS.
- Стоп-слово: отдельный keyword detector, обучаемый на `processed_dataset/stop_*.wav`.

## Быстрый запуск на ПК

```bash
python3.13 -m venv .venv313
source .venv313/bin/activate
pip install -r requirements.txt

python scripts/train_stop_detector.py
python scripts/download_models.py
bash scripts/build_llama_cpp.sh auto
bash scripts/start_llama_server.sh
```

Если Hugging Face качает GGUF медленно или оборвал соединение, просто повторите `python scripts/download_models.py --skip-vosk`: загрузка продолжится. `start_llama_server.sh` сначала ищет локальные файлы в `models/qwen2_5_vl_3b_gguf`, а если их еще нет, запускает `llama-server -hf` и дает `llama.cpp` скачать модель в HF cache.

На macOS для `Qwen2.5-VL` скрипт по умолчанию включает безопасный режим `CPU-only`, потому что в текущей сборке `llama.cpp` мультимодальный `llama-server` с Metal может падать на `GGML_ASSERT(buf_dst) failed` во время инициализации слотов. Это медленнее, но реально работает. Если хотите снова попробовать Metal, запускайте так:

```bash
FORCE_UNSAFE_METAL=1 bash scripts/start_llama_server.sh
```

В другом терминале:

```bash
source .venv313/bin/activate
python -m vision_voice_assistant \
  --camera laptop \
  --voice \
  --tts system
```

В окне камеры: `Space` отправляет кадр в модель, `q` закрывает программу. Голосом можно сказать: "опиши", "что видишь", "запомни ...", "что ты помнишь", "стоп", "выход".

Для встроенной камеры ноутбука можно явно выбрать индекс:

```bash
python -m vision_voice_assistant --camera laptop --camera-index 0 --voice --tts system
```

Если встроенная камера не открылась, попробуйте `--camera-index 1`.

## Запуск с vLLM

На CUDA-машине или в Colab:

```bash
pip install -r requirements-colab.txt
bash scripts/start_vllm_server.sh
```

Ассистент:

```bash
python -m vision_voice_assistant \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --camera http://PHONE_IP:8080/shot.jpg \
  --no-preview \
  --voice \
  --tts none
```

Подробности для Colab лежат в `docs/COLAB.md`.

## Телефон как камера

Подойдет любое приложение, которое отдает MJPEG или snapshot по HTTP:

- IP Webcam: `http://PHONE_IP:8080/video` или `http://PHONE_IP:8080/shot.jpg`
- DroidCam: часто `http://PHONE_IP:4747/video`
- Iriun / OBS / NDI можно подключить как обычную локальную камеру `--camera 0`

## Камера ноутбука

Самый простой локальный запуск теперь такой:

```bash
python -m vision_voice_assistant --camera laptop --voice --tts system
```

Без аргумента `--camera` программа тоже по умолчанию возьмет локальную камеру `0`.

## Проверка без камеры

```bash
python scripts/smoke_test.py
python -m vision_voice_assistant --image path/to/photo.jpg --no-tts
```

## Датасет для всего проекта

Для проекта теперь есть отдельный reproducible dataset pipeline:

```bash
python scripts/assemble_project_dataset.py all
```

Он собирает:

- локальные `stop` positives
- русскую речь для STT и speech negatives
- шумовые negatives
- vision description eval
- OCR eval на `COCO-Text` с автоматической докачкой нужного subset изображений
- memory regression scenarios

Подробности и оговорки по лицензиям: `docs/DATASET_GUIDE.md`.

## Проверенный локальный режим на Mac

На Apple Silicon у меня успешно отработал такой путь:

1. `bash scripts/start_llama_server.sh`  
   сервер поднялся в `CPU-only` safe mode и отвечал на `http://127.0.0.1:8080/v1`

2. тестовый запрос с картинкой через `vision_voice_assistant.vlm_client.OpenAICompatibleVLM`  
   прошел без падения

Это значит, что весь проект можно гонять локально на Mac, но ждать ответа от VLM придется заметно дольше, чем на GPU/Colab.

## Почему такая модель

Базовый локальный вариант: `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q4_K_M`. Она маленькая для VLM, скачивается `llama.cpp` через `-hf`, а `mmproj` подтягивается автоматически, если доступен. Для Colab/GPU используется исходная HF-модель `Qwen/Qwen2.5-VL-3B-Instruct` через `vLLM`.

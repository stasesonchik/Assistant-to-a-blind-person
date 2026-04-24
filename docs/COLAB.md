# Colab / GPU запуск

Этот вариант нужен, если на ПК нет нормальной GPU или хочется поднять VLM через `vLLM`.

1. Скопируйте проект в Colab или подключите Google Drive.

2. Установите зависимости:

```bash
pip install -r requirements-colab.txt
python scripts/train_stop_detector.py
python scripts/download_models.py --skip-gguf
```

3. Поднимите сервер VLM:

```bash
bash scripts/start_vllm_server.sh
```

4. В другой ячейке запустите ассистента. В Colab чаще всего удобнее использовать телефон как IP-камеру и режим без окна:

```bash
python -m vision_voice_assistant \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --camera-url http://PHONE_IP:8080/shot.jpg \
  --no-preview \
  --voice \
  --tts none
```

Для телефона проще всего поставить приложение IP Webcam / DroidCam / Iriun и взять URL snapshot или MJPEG. Для IP Webcam обычно подходят:

```text
http://PHONE_IP:8080/video
http://PHONE_IP:8080/shot.jpg
```

Если микрофон Colab недоступен, голосовую часть лучше проверять на ПК. VLM-сервер при этом можно оставить в Colab и пробросить адрес через ngrok/cloudflared.


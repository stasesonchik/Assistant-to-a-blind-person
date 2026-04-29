# Dataset Guide

Этот файл описывает датасет-пакет для всего проекта: `stop` keyword spotting, STT/голосовые команды, vision description, OCR и memory regression.

## Цель

Нам нужен не один “магический” датасет, а воспроизводимый набор из нескольких источников:

- `stop` positives: ваши локальные записи слова `стоп`
- speech negatives + STT eval: реальные русские записи с разметкой
- noise negatives: фоновые и бытовые звуки
- vision description eval: реальные object-centric изображения с captions/object labels
- OCR eval: изображения с текстом в сцене
- memory eval: сценарии “посмотри -> запомни -> ответь позже”

## Что выбрано

### Базовый набор

1. `processed_dataset/stop_*.wav`
   - Локальные положительные примеры `стоп`
   - Уже есть в проекте
   - Статус: `DONE`

2. `OpenSTT` validation subsets
   - `asr_calls_2_val`
   - `buriy_audiobooks_2_val`
   - `public_youtube700_val`
   - Зачем: реальные русские негативы для keyword spotting и готовый STT eval
   - Лицензия: `CC BY-NC 4.0`, коммерческое использование требует отдельного согласования
   - Статус: `AUTOMATED`

3. `ESC-50`
   - Зачем: hard negatives по шумам и бытовым звукам
   - Лицензия: `CC BY-NC 3.0`, subset `ESC-10` — `CC BY`
   - Статус: `AUTOMATED`

4. `COCO val2017 + annotations`
   - Зачем: vision description eval, object labels, memory scenarios
   - Статус: `AUTOMATED`

5. `COCO-Text + selected COCO train2014 image subset`
   - Зачем: OCR/scene text eval
   - Статус: `AUTOMATED`

### Дополнительный набор

6. `MUSAN`
   - Зачем: большой корпус noise/music/speech для более жестких негативов
   - Минус: большой объем
   - Статус: `OPTIONAL`

## Почему именно так

- `OpenSTT` дает нам русскую речь с manual annotations, а не синтетику.
- `ESC-50` быстро закрывает бытовые шумы, которые часто ломают keyword spotting.
- `COCO val2017` хорошо подходит для оценки описания предметов и сцены.
- `COCO-Text` закрывает режим “прочитай, что написано”.
- Для него не нужно скачивать весь `train2014.zip`: скрипт автоматически забирает только нужный subset изображений.
- `memory_eval` собирается поверх COCO автоматически: парные кадры + expected objects с первого кадра.

## Что собирает скрипт

Главный скрипт:

```bash
python scripts/assemble_project_dataset.py all
```

Он:

1. скачивает выбранные источники в `data/project_dataset/downloads`
2. распаковывает их в `data/project_dataset/raw`
3. для `COCO-Text` отдельно докачивает только нужные `train2014`-изображения
4. строит manifests в `data/project_dataset/manifests`
5. пишет отчеты в `data/project_dataset/reports`

Большие загрузки поддерживают возобновление по `.part` файлам: если сеть оборвется или процесс прервется, повторный запуск продолжит скачивание с места.

### Полезные команды

Показать источники:

```bash
python scripts/assemble_project_dataset.py --list-sources
```

Скачать только базовый набор:

```bash
python scripts/assemble_project_dataset.py download
```

Скачать базовый набор + MUSAN:

```bash
python scripts/assemble_project_dataset.py download --include-musan
```

Построить manifests заново без скачивания:

```bash
python scripts/assemble_project_dataset.py build
```

## Итоговые manifests

После сборки появятся:

- `kws_positive_local.csv`
- `kws_negative_speech.csv`
- `kws_negative_noise.csv`
- `kws_eval.csv`
- `stt_eval_openstt.csv`
- `voice_commands_ru.csv`
- `voice_command_hits_openstt.csv`
- `vision_description_coco_val2017.csv`
- `vision_ocr_cocotext_val2014.csv`
- `manifest_validation.json`
- `memory_eval_coco.jsonl`

## Что уже считается завершенным

- `DONE`: локальные positive examples `стоп`
- `DONE`: структура dataset pipeline в коде
- `DONE`: автоматическая сборка manifests
- `DONE`: подготовка отдельного memory regression set

## Что остается проверить после сборки

1. Прогнать `kws_eval.csv` на текущем стоп-детекторе и измерить false positives.
2. Проверить `voice_command_hits_openstt.csv`: хватает ли реальных command-like фраз.
3. Прогнать `vision_description_coco_val2017.csv` через ваш VLM.
4. Прогнать `vision_ocr_cocotext_val2014.csv` в режиме чтения текста.
5. Прогнать `memory_eval_coco.jsonl` как регрессионный набор для памяти.
6. Проверить `reports/manifest_validation.json`: там видно, что пути из manifests реально существуют.

## Важная оговорка по лицензиям

Базовый пакет практичный для учебного/исследовательского проекта, но не весь одинаково хорош для коммерческого использования:

- `OpenSTT`: `CC BY-NC 4.0`
- `ESC-50`: `CC BY-NC 3.0`

Если нужен коммерчески более чистый стек, добавляйте `MUSAN` и отдельно подбирайте permissive русскую ASR-оценку.

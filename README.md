# Сверка Go API

«Сверка» принимает PDF со статьями о RAG, сохраняет оригиналы и обрабатывает задания из SQLite. API возвращает результаты, связанные сущности, указатели на источник и замечания. По умолчанию работает синтетический извлекатель для разработки фронтенда.

Межстатейное сравнение следует правилам RAG-кейса: сопоставляются задача, датасет, определение метрики, шкала и протокол оценки; разные конфигурации остаются отдельными результатами. Раздел `#groups` собирает только подтверждённые сопоставимые результаты, граф связей между статьями не строится.

## Запуск

Нужен Go 1.26. Из каталога `sverka` выполните:

```bash
go run ./cmd/api
```

Если Go установлен в `/usr/local/go`, добавьте `/usr/local/go/bin` в `PATH`. Откройте `http://localhost:8080/`, чтобы загрузить статью, перейти к источнику и сравнить результаты. Интерфейс и API обслуживает один Go-процесс, который хранит базу `sverka.sqlite` и оригиналы PDF в каталоге `data/`. Ранее созданный файл SQLite подхватывается автоматически, если в каталоге ещё нет `sverka.sqlite`.

Конфигурация задаётся переменными окружения. Значения по умолчанию перечислены в [.env.example](.env.example). Для фронта с другим адресом укажите его точный origin в `CORS_ORIGIN`. Лимит загрузки по умолчанию — 32 МиБ.

## Проверка по HTTP

```bash
api=http://localhost:8080/api/v1
pdf=/path/to/D019.pdf

doc=$(curl -fsS -F "file=@$pdf" "$api/documents" | jq -r .document_id)
curl -fsS "$api/documents/$doc" | jq '.document | {status, extraction_mode, failure}'
```

После `completed` выполните:

```bash
results=$(curl -fsS "$api/results?document_id=$doc&page=1")
left=$(jq -r '.items[0].result.result_id' <<<"$results")
right=$(jq -r '.items[1].result.result_id' <<<"$results")
curl -fsS "$api/results/$left" | jq
curl -fsS -H 'Content-Type: application/json' \
  --data "$(jq -cn --arg left "$left" --arg right "$right" \
         '{left_result_id:$left,right_result_id:$right}')" "$api/comparisons" | jq
curl -fsS "$api/export?document_id=$doc" | jq
```

Mock возвращает синтетические результаты независимо от содержания PDF. Фронт помечает их по `extraction_mode`. В JSON-экспорт входят метаданные документов и результаты их обработки. Маппинг для рекомендованной выгрузки в семь CSV описан в контракте извлекателя.

## Подключение Python

Встроенный извлекатель запускается на Python 3.10+ и использует утилиты Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`). Для сканов установите Tesseract с языком `eng`. На Debian и Ubuntu это пакеты `poppler-utils`, `tesseract-ocr` и `tesseract-ocr-eng`. Путь к Tesseract можно задать через `TESSERACT_BIN`, язык — через `OCR_LANG`.

```bash
python3 extractor/main.py --pdf /path/to/D019.pdf | jq '.results | length'
EXTRACTION_MODE=python PYTHON_SCRIPT=extractor/main.py go run ./cmd/api
```

Go передаёт `--pdf <абсолютный путь>` при запуске `PYTHON_BIN`, ждёт JSON из stdout и ограничивает время через `EXTRACT_TIMEOUT`. Режим сохраняется в каждом задании. Сбой Python виден в статусе документа и журнале сервера.

Извлекатель обрабатывает таблицы с распознаваемыми заголовками метрик и строками чисел. Сохраняются исходное значение, номер страницы, таблица, строка и столбец. Величины из графиков, произвольные формулировки результатов и полное покрытие корпуса требуют отдельной проверки. Если Tesseract недоступен, страницы без текста отмечаются как непрочитанные. Полностью непрочитанный PDF завершает задачу с ошибкой.

## Пакетная выгрузка

Команда `python3 -m batch` загружает PDF корпуса через API и формирует семь CSV в формате шаблона организаторов. Она также умеет возобновить обработку по сохранённым заданиям и преобразовать скачанный JSON офлайн. [Инструкция по пакетной обработке](batch/README.md)

## Запуск в контейнере

```bash
docker compose up --build -d --wait
```

Контейнер запускает Python-извлечение с OCR и хранит базу и оригиналы в именованном томе. [Конфигурация и эксплуатация](docs/deploy.md)

## Защита проекта

Краткий питч, маршрут демонстрации и измеренные ограничения: [docs/защита.md](docs/защита.md).
План параллельной работы трёх участников и владение файлами: [docs/team-workflow.md](docs/team-workflow.md).

## Контракты и проверки

- [OpenAPI](docs/openapi.yaml) и [интеграция фронтенда](docs/frontend-contract.md)
- [Контракт Python](docs/extractor-contract.md) и [JSON Schema](docs/extractor.schema.json)
- [Паспорт данных](docs/data-schema.md)

```bash
go test ./...
go vet ./...
python3 -m unittest discover -s extractor
python3 -m unittest discover -s batch
```

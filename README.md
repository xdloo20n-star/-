# Сверка Go API

«Сверка» принимает PDF со статьями о RAG, сохраняет оригиналы и обрабатывает задания из SQLite. API возвращает результаты, связанные сущности, указатели на источник и замечания. По умолчанию работает синтетический извлекатель для разработки фронтенда.

## Запуск

Нужен Go 1.26. Из каталога `sverka` выполните:

```bash
go run ./cmd/api
```

Если Go установлен в `/usr/local/go`, добавьте `/usr/local/go/bin` в `PATH`. Локальный API доступен по `http://localhost:8080`, здоровье — по `/healthz`. Один экземпляр API использует каталог данных `data/` с базой `sverka.sqlite` и оригиналами в `pdf/`. Ранее созданный файл SQLite подхватывается автоматически, если в каталоге ещё нет `sverka.sqlite`.

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

```bash
EXTRACTION_MODE=python PYTHON_SCRIPT=/absolute/path/to/extractor.py go run ./cmd/api
```

Go передаёт `--pdf <абсолютный путь>` при запуске `PYTHON_BIN`, ждёт JSON из stdout и ограничивает время через `EXTRACT_TIMEOUT`. Режим сохраняется в каждом задании. Сбой Python виден в статусе документа и журнале сервера.

## Контракты и проверки

- [OpenAPI](docs/openapi.yaml) и [интеграция фронтенда](docs/frontend-contract.md)
- [Контракт Python](docs/extractor-contract.md) и [JSON Schema](docs/extractor.schema.json)
- [Паспорт данных](docs/data-schema.md)

```bash
go test ./...
go vet ./...
```

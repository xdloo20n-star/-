FROM golang:1.26.5-trixie AS build
WORKDIR /src
ENV GOTOOLCHAIN=local

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o /out/sverka-api ./cmd/api

FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        fonts-dejavu-core \
        poppler-utils \
        python3 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 sverka \
    && useradd --uid 10001 --gid 10001 --no-create-home --no-log-init sverka \
    && install -d -m 0750 -o 10001 -g 10001 /data

WORKDIR /app
COPY --from=build /out/sverka-api /usr/local/bin/sverka-api
COPY extractor/ /app/extractor/

ENV HOST=0.0.0.0 \
    PORT=8080 \
    DATA_DIR=/data \
    EXTRACTION_MODE=python \
    PYTHON_BIN=python3 \
    PYTHON_SCRIPT=/app/extractor/main.py \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSERACT_BIN=tesseract \
    OCR_LANG=eng

USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error --output /dev/null --noproxy '*' \
        --max-time 3 "http://127.0.0.1:${PORT}/healthz" || exit 1
ENTRYPOINT ["/usr/local/bin/sverka-api"]

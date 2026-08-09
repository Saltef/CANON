FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CANON_SETTINGS=/app/conf/settings.toml
ENV CANON_DATA_DIR=/app/data
ENV CANON_REPORTS_DIR=/app/reports
ENV PORT=8000

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY canon /app/canon
COPY conf /app/conf
COPY data /app/data
COPY docs /app/docs
COPY gold /app/gold

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[serve,vectorstores,docs]" \
    && mkdir -p /app/data/raw /app/data/processed /app/reports \
    && mkdir -p /app/storage/data/raw /app/storage/data/processed /app/storage/reports

EXPOSE 8000

CMD ["sh", "-c", "python -m canon.product.asgi --host 0.0.0.0 --port ${PORT:-8000} --max-concurrency ${CANON_MAX_CONCURRENCY:-8} --max-queue-depth ${CANON_MAX_QUEUE_DEPTH:-16}"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY canon /app/canon
COPY conf /app/conf
COPY data /app/data
COPY docs /app/docs
COPY gold /app/gold
COPY tests /app/tests

CMD ["python", "-m", "canon.product.server", "--host", "0.0.0.0", "--port", "8000"]

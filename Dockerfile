FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-local-llm.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt -r requirements-local-llm.txt

COPY app ./app
COPY run_api.py ./run_api.py

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/models \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["python", "run_api.py"]

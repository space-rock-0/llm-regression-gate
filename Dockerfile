# MRDS — Model Regression Detection System (container image)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROVIDER=mock \
    OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
    WARN_THRESHOLD=0.03 \
    CRITICAL_THRESHOLD=0.08

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code, prompts, golden dataset, runner, report generator.
COPY . .

# Default: fast mock evaluation (no live LLM).
CMD ["python", "runner.py", "--provider", "mock", "--no-judge", "--concurrency", "8"]
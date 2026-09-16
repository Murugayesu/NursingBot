FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# ── App source ───────────────────────────────────────────────────────────────
COPY . .

# ── Run migrations then start API ────────────────────────────────────────────
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]

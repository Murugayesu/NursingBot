FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (#7) ────────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# ── App source ────────────────────────────────────────────────────────────────
COPY --chown=appuser:appgroup . .

# ── Switch to non-root (#7) ───────────────────────────────────────────────────
USER appuser

# ── Healthcheck (#21) ─────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
# NOTE: Migrations are intentionally NOT run here (#5).
# Run `alembic upgrade head` once as a separate release step before deploying.
# This avoids races when scaling to multiple replicas.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

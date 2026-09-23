## What and why

## Checklist
- [ ] `make check` passes (ruff, mypy, unit tests)
- [ ] Schema change? Added an Alembic migration and confirmed `alembic upgrade head` on a fresh DB
- [ ] New env var? Added to `.env.example` and `app/config/settings.py`
- [ ] Touches upload, auth, or tenant scoping? Called out for security review

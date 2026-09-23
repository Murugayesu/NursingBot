.PHONY: venv up down migrate revision lint fmt typecheck test test-int check

# A project venv avoids other editable installs shadowing the `app` package.
venv:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

up:
	docker compose up -d

down:
	docker compose down

migrate:
	docker compose exec api alembic upgrade head

# usage: make revision m="add foo column"
revision:
	alembic revision --autogenerate -m "$(m)"

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

typecheck:
	mypy -p app --ignore-missing-imports

test:
	pytest tests/unit -q

test-int:
	pytest tests/integration -q

check: lint typecheck test

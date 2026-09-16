from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_health_endpoint_exists(client):
    """Health endpoint should return 200 even if services are down (returns degraded)."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


@pytest.mark.asyncio
async def test_root_endpoint(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "RAG Platform"


@pytest.mark.asyncio
async def test_openapi_schema(client):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    # Verify key endpoints are documented
    assert "/knowledge-bases" in schema["paths"]
    assert "/health" in schema["paths"]

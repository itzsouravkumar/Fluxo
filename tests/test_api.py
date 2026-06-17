"""Tests for FLUXO API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_health():
    from core.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/junctions/")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_junctions_list():
    from core.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/junctions/")
        assert response.status_code == 200
        data = response.json()
        assert "junctions" in data


@pytest.mark.asyncio
async def test_violations_list():
    from core.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/violations/")
        assert response.status_code == 200

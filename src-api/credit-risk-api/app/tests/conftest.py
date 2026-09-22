from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


# Cliente de prueba
@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}


@pytest.fixture()
def solicitud_bajo_riesgo() -> dict:
    return {
        "utilizacion": 0.1,
        "edad": 45,
        "moras_30_59": 0,
        "razon_deuda": 0.2,
        "ingreso_mensual": 8000,
        "lineas_abiertas": 10,
        "moras_90": 0,
        "creditos_inmob": 1,
        "moras_60_89": 0,
        "dependientes": 1,
    }


@pytest.fixture()
def solicitud_alto_riesgo() -> dict:
    return {
        "utilizacion": 0.98,
        "edad": 24,
        "moras_30_59": 4,
        "razon_deuda": 1.5,
        "ingreso_mensual": None,
        "lineas_abiertas": 2,
        "moras_90": 3,
        "creditos_inmob": 0,
        "moras_60_89": 2,
        "dependientes": 0,
    }

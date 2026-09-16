import io
import sqlite3

import pytest
from PIL import Image

import app as app_module
from app import CLASS_NAMES, create_app


@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client


def test_homepage_renders(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Crop Leaf Disease Identifier' in response.data
    assert b'Analyze a Leaf' in response.data


def test_disease_information_pages_render(client):
    assert client.get('/diseases').status_code == 200
    detail = client.get('/diseases/Leaf-Spot')
    assert detail.status_code == 200
    assert b'Leaf Spot' in detail.data


def test_disease_classes_are_expanded():
    assert len(CLASS_NAMES) >= 7
    assert 'Late Blight' in CLASS_NAMES
    assert 'Mosaic Virus' in CLASS_NAMES


def test_api_health_and_disease_routes(client):
    health = client.get('/api/health')
    assert health.status_code == 200
    payload = health.get_json()
    assert payload['status'] == 'ok'
    assert 'classes' in payload

    diseases = client.get('/api/diseases')
    assert diseases.status_code == 200
    assert isinstance(diseases.get_json(), list)


def test_predict_endpoint_accepts_image(client):
    buffer = io.BytesIO()
    Image.new('RGB', (64, 64), color='green').save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    response = client.post(
        '/api/predict',
        data={'file': (io.BytesIO(image_bytes), 'leaf.png')},
        content_type='multipart/form-data',
    )
    if app_module.app_model_cache is None:
        assert response.status_code == 503
        assert response.get_json()['code'] == 'model_unavailable'
        assert response.get_json()['success'] is False
        return

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['success'] is True
    assert 'prediction' in payload
    assert 'confidence' in payload
    assert 'disease_info' in payload

    history = client.get('/api/history')
    assert history.status_code == 200
    assert isinstance(history.get_json(), list)
    assert history.get_json()[0]['prediction'] == payload['prediction']


def test_invalid_upload_is_rejected(client):
    response = client.post(
        '/api/predict',
        data={'file': (io.BytesIO(b'not-an-image'), 'note.txt')},
        content_type='multipart/form-data',
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert 'error' in payload


def test_history_keeps_only_latest_five(monkeypatch, tmp_path):
    db_path = tmp_path / 'predictions.db'
    monkeypatch.setattr(app_module, 'DB_PATH', db_path)

    app_module._init_db()
    for index in range(10):
        app_module._log_prediction(
            uploaded_name=f'scan-{index}.png',
            saved_image_path=f'uploads/scan-{index}.png',
            label='Healthy',
            confidence=0.90,
            crop='General',
            severity='Low',
            urgency='Low',
            disease_info={'crop': 'General'},
            top_predictions=[{'label': 'Healthy', 'probability': 90.0}],
        )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        ids = conn.execute('SELECT id FROM predictions ORDER BY id ASC').fetchall()

    assert rows == 5
    assert len(ids) == 5
    assert ids[0][0] >= 6

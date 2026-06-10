"""Tests for Exoplanet AI Hunter API."""
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'web_app'))

import pytest
from app import app, load_model


@pytest.fixture(scope='session', autouse=True)
def setup_model():
    load_model()


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


EXAMPLE_INPUT = {
    'orbital_period': 9.49, 'transit_duration': 2.96, 'planetary_radius': 2.26,
    'transit_depth': 615.8, 'impact_parameter': 0.15,
    'equilibrium_temperature': 793, 'insolation_flux': 105.0,
    'stellar_surface_gravity': 4.47, 'stellar_radius': 0.93
}


def test_index(client):
    r = client.get('/')
    assert r.status_code == 200
    assert b'Exoplanet AI Hunter' in r.data


def test_predict_full_features(client):
    r = client.post('/api/predict', json=EXAMPLE_INPUT)
    assert r.status_code == 200
    data = r.get_json()
    assert data['disposition'] in ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
    assert 0 <= data['confidence'] <= 1
    assert 'all_probabilities' in data


def test_predict_minimal_features(client):
    r = client.post('/api/predict', json={
        'orbital_period': 9.49, 'transit_duration': 2.96, 'planetary_radius': 2.26,
        'equilibrium_temperature': 793, 'stellar_radius': 0.93
    })
    assert r.status_code == 200
    data = r.get_json()
    assert data['disposition'] in ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']


def test_predict_insufficient_data(client):
    r = client.post('/api/predict', json={'orbital_period': 5.0})
    assert r.status_code == 200
    data = r.get_json()
    assert data['disposition'] == 'NO_PREDICT'


def test_feature_importance(client):
    r = client.get('/api/feature-importance')
    assert r.status_code == 200
    data = r.get_json()
    assert len(data['feature_importance']) == 9


def test_model_info(client):
    r = client.get('/api/model-info')
    assert r.status_code == 200
    data = r.get_json()
    assert data['model_type'] == 'random_forest'
    assert len(data['classes']) == 3

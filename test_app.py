# test_app.py — Basic tests for Disease Prediction Flask App
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test 1 — Home page loads correctly
def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    print("✅ Test 1 passed — Home page loads!")

# Test 2 — Valid input returns prediction
def test_predict_valid_input(client):
    response = client.post('/predict', data={
        'pregnancies':    '2',
        'glucose':        '130',
        'blood_pressure': '70',
        'skin_thickness': '30',
        'insulin':        '100',
        'bmi':            '28.5',
        'dpf':            '0.45',
        'age':            '25',
        'disclaimer':     'on'
    })
    assert response.status_code == 200
    # Check result is shown
    assert b'Risk' in response.data
    print("✅ Test 2 passed — Valid input gives prediction!")

# Test 3 — Missing fields shows error
def test_predict_missing_field(client):
    response = client.post('/predict', data={
        'pregnancies': '2'
        # all other fields missing
    })
    assert response.status_code == 200
    assert b'fill' in response.data or b'valid' in response.data or b'Please' in response.data
    print("✅ Test 3 passed — Missing fields handled correctly!")

# Test 4 — Invalid glucose value shows error
def test_predict_invalid_glucose(client):
    response = client.post('/predict', data={
        'pregnancies':    '2',
        'glucose':        '999',  # out of range
        'blood_pressure': '70',
        'skin_thickness': '30',
        'insulin':        '100',
        'bmi':            '28.5',
        'dpf':            '0.45',
        'age':            '25',
        'disclaimer':     'on'
    })
    assert response.status_code == 200
    print("✅ Test 4 passed — Invalid glucose handled!")

# Test 5 — Health endpoint works
def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    print("✅ Test 5 passed — Health endpoint works!")
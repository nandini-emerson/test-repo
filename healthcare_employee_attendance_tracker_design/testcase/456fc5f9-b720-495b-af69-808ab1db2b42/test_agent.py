
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask, jsonify, request
import json

@pytest.fixture
def app():
    """
    Fixture to create a Flask app instance for testing.
    """
    app = Flask(__name__)

    @app.route('/attendance/record', methods=['POST'])
    def record_attendance():
        auth_header = request.headers.get('Authorization')
        if not auth_header or auth_header != "Bearer valid_token":
            return jsonify({
                "success": False,
                "status": "error",
                "message": "Authentication failed"
            }), 401

        data = request.get_json()
        employee_id = data.get('employee_id')
        shift_id = data.get('shift_id')
        date = data.get('date')

        # Simulate HRIS API call and validation
        if employee_id == "INVALID":
            return jsonify({
                "success": False,
                "status": "error",
                "message": "Invalid employee_id"
            }), 400
        if shift_id == "INVALID":
            return jsonify({
                "success": False,
                "status": "error",
                "message": "Invalid shift assignment"
            }), 400
        if employee_id == "HRIS_FAIL":
            return jsonify({
                "success": False,
                "status": "error",
                "message": "HRIS API failure"
            }), 502

        return jsonify({
            "success": True,
            "status": "success",
            "message": f"Attendance for employee {employee_id} recorded successfully"
        }), 200

    return app

@pytest.fixture
def client(app):
    """
    Fixture to provide a test client for the Flask app.
    """
    return app.test_client()

def test_record_attendance_functional_success(client):
    """
    Functional test: Validates that the /attendance/record endpoint successfully records attendance
    for a valid employee, shift, and date with proper authentication.
    """
    payload = {
        "employee_id": "EMP1234",
        "date": "2024-06-01",
        "shift_id": "SHIFT001"
    }
    headers = {
        "Authorization": "Bearer valid_token",
        "Content-Type": "application/json"
    }
    response = client.post('/attendance/record', data=json.dumps(payload), headers=headers)
    assert response.status_code == 200, "Expected HTTP 200 for successful attendance record"
    resp_json = response.get_json()
    assert resp_json['success'] is True, "Expected 'success' field to be True"
    assert resp_json['status'] == 'success', "Expected 'status' field to be 'success'"
    assert f"Attendance for employee {payload['employee_id']}" in resp_json['message'], \
        "Expected confirmation message containing employee id"

@pytest.mark.parametrize("payload,headers,expected_status,expected_message", [
    ({"employee_id": "INVALID", "date": "2024-06-01", "shift_id": "SHIFT001"},
     {"Authorization": "Bearer valid_token", "Content-Type": "application/json"},
     400, "Invalid employee_id"),
    ({"employee_id": "EMP1234", "date": "2024-06-01", "shift_id": "INVALID"},
     {"Authorization": "Bearer valid_token", "Content-Type": "application/json"},
     400, "Invalid shift assignment"),
    ({"employee_id": "EMP1234", "date": "2024-06-01", "shift_id": "SHIFT001"},
     {"Authorization": "Bearer invalid_token", "Content-Type": "application/json"},
     401, "Authentication failed"),
    ({"employee_id": "HRIS_FAIL", "date": "2024-06-01", "shift_id": "SHIFT001"},
     {"Authorization": "Bearer valid_token", "Content-Type": "application/json"},
     502, "HRIS API failure"),
])
def test_record_attendance_functional_errors(client, payload, headers, expected_status, expected_message):
    """
    Functional test: Validates error scenarios for /attendance/record endpoint:
    - Invalid employee_id
    - Invalid shift assignment
    - Authentication failure
    - HRIS API failure
    """
    response = client.post('/attendance/record', data=json.dumps(payload), headers=headers)
    assert response.status_code == expected_status, f"Expected HTTP {expected_status} for error scenario"
    resp_json = response.get_json()
    assert resp_json['success'] is False, "Expected 'success' field to be False"
    assert resp_json['status'] == 'error', "Expected 'status' field to be 'error'"
    assert expected_message in resp_json['message'], f"Expected error message '{expected_message}'"


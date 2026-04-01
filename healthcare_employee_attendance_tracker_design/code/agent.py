
import os
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, constr
from dotenv import load_dotenv
import openai
import requests
from loguru import logger
from cachetools import TTLCache
from cryptography.fernet import Fernet
from jinja2 import Template
import pandas as pd

# =========================
# Configuration Management
# =========================

class Config:
    """Centralized configuration management."""
    load_dotenv()
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    HRIS_API_URL: str = os.getenv("HRIS_API_URL", "")
    HRIS_API_TOKEN: str = os.getenv("HRIS_API_TOKEN", "")
    REPORT_API_URL: str = os.getenv("REPORT_API_URL", "")
    REPORT_API_KEY: str = os.getenv("REPORT_API_KEY", "")
    NOTIFICATION_API_URL: str = os.getenv("NOTIFICATION_API_URL", "")
    NOTIFICATION_API_KEY: str = os.getenv("NOTIFICATION_API_KEY", "")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_TEXT_LENGTH: int = 50000

    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.HRIS_API_URL:
            missing.append("HRIS_API_URL")
        if not cls.HRIS_API_TOKEN:
            missing.append("HRIS_API_TOKEN")
        if not cls.REPORT_API_URL:
            missing.append("REPORT_API_URL")
        if not cls.REPORT_API_KEY:
            missing.append("REPORT_API_KEY")
        if not cls.NOTIFICATION_API_URL:
            missing.append("NOTIFICATION_API_URL")
        if not cls.NOTIFICATION_API_KEY:
            missing.append("NOTIFICATION_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required configuration: {', '.join(missing)}")

Config.validate()

# =========================
# Logging Configuration
# =========================

logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=Config.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    backtrace=True,
    diagnose=True,
)

# =========================
# Utility Functions
# =========================

def mask_pii(text: str) -> str:
    """Mask employee IDs and other PII in logs and outputs."""
    return re.sub(r"\b\d{4,}\b", "****", text)

def redact_sensitive(data: Any) -> Any:
    """Recursively mask PII in dicts/lists."""
    if isinstance(data, dict):
        return {k: redact_sensitive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_sensitive(i) for i in data]
    elif isinstance(data, str):
        return mask_pii(data)
    return data

def encrypt_data(data: str) -> str:
    """Encrypt data using AES-256 (Fernet)."""
    f = Fernet(Config.ENCRYPTION_KEY.encode())
    return f.encrypt(data.encode()).decode()

def decrypt_data(token: str) -> str:
    """Decrypt data using AES-256 (Fernet)."""
    f = Fernet(Config.ENCRYPTION_KEY.encode())
    return f.decrypt(token.encode()).decode()

def exponential_backoff(retries: int) -> float:
    return min(2 ** retries, 30)

# =========================
# Pydantic Models
# =========================

class AttendanceInput(BaseModel):
    employee_id: constr(strip_whitespace=True, min_length=3, max_length=32)
    date: constr(strip_whitespace=True, pattern=r"^\d{4}-\d{2}-\d{2}$")
    shift_id: constr(strip_whitespace=True, min_length=1, max_length=16)

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        if not re.match(r"^\d{3,}$", v):
            raise ValueError("Employee ID must be numeric and at least 3 digits.")
        return v

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        # Simple YYYY-MM-DD check is done by regex, further validation can be added
        return v

    @field_validator("shift_id")
    @classmethod
    def validate_shift_id(cls, v):
        if not v:
            raise ValueError("Shift ID cannot be empty.")
        return v

class ReportInput(BaseModel):
    date_range: constr(strip_whitespace=True, min_length=9, max_length=23)
    department: constr(strip_whitespace=True, min_length=2, max_length=64)

    @field_validator("date_range")
    @classmethod
    def validate_date_range(cls, v):
        # Accepts "YYYY-MM-DD to YYYY-MM-DD"
        if not re.match(r"^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date range must be in format 'YYYY-MM-DD to YYYY-MM-DD'.")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v):
        if not v.isalpha() and not all(x.isalpha() or x.isspace() for x in v):
            raise ValueError("Department must contain only letters and spaces.")
        return v

class NotificationInput(BaseModel):
    employee_id: constr(strip_whitespace=True, min_length=3, max_length=32)
    message: constr(strip_whitespace=True, min_length=1, max_length=500)

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        if not re.match(r"^\d{3,}$", v):
            raise ValueError("Employee ID must be numeric and at least 3 digits.")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty.")
        return v.strip()

class AnomalyInput(BaseModel):
    date_range: constr(strip_whitespace=True, min_length=9, max_length=23)
    department: constr(strip_whitespace=True, min_length=2, max_length=64)

    @field_validator("date_range")
    @classmethod
    def validate_date_range(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date range must be in format 'YYYY-MM-DD to YYYY-MM-DD'.")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v):
        if not v.isalpha() and not all(x.isalpha() or x.isspace() for x in v):
            raise ValueError("Department must contain only letters and spaces.")
        return v

# =========================
# Base Agent
# =========================

class BaseAgent:
    """Generic base agent class."""
    def __init__(self):
        pass

# =========================
# Input Processor
# =========================

class InputProcessor:
    """Parses and validates user input, extracts required parameters."""

    def parse_input(self, data: dict, input_type: str) -> Any:
        """Parse and validate input based on type."""
        try:
            if input_type == "attendance":
                return AttendanceInput(**data)
            elif input_type == "report":
                return ReportInput(**data)
            elif input_type == "notification":
                return NotificationInput(**data)
            elif input_type == "anomaly":
                return AnomalyInput(**data)
            else:
                raise ValueError("Unknown input type.")
        except ValidationError as ve:
            logger.error(f"Input validation error: {ve}")
            raise

    def validate_input(self, model: BaseModel) -> bool:
        """Additional validation if needed."""
        # All validation is handled by Pydantic models
        return True

# =========================
# Authentication Service
# =========================

class AuthenticationService:
    """Handles SSO and multi-factor authentication, session management."""

    def __init__(self):
        self.session_cache = TTLCache(maxsize=1000, ttl=900)  # 15 min session

    def authenticate_user(self, token: str) -> bool:
        """Validate user session token (stub for SSO/MFA)."""
        # In production, validate token with SSO provider
        if not token or not isinstance(token, str):
            return False
        # Simulate session check
        if token in self.session_cache:
            return True
        # Simulate token validation with HRIS
        headers = {"Authorization": f"Bearer {Config.HRIS_API_TOKEN}"}
        try:
            resp = requests.get(f"{Config.HRIS_API_URL}/validate_session", headers=headers, params={"token": token}, timeout=3)
            if resp.status_code == 200 and resp.json().get("valid"):
                self.session_cache[token] = True
                return True
        except Exception as e:
            logger.error(f"Auth service error: {e}")
        return False

    def authorize_action(self, user_id: str, action: str) -> bool:
        """Check if user is authorized for the action."""
        # In production, check user roles/permissions via HRIS
        # Here, allow all actions for demo
        return True

# =========================
# Domain Logic Engine
# =========================

class DomainLogicEngine:
    """Implements business rules for attendance validation, shift checks, anomaly detection."""

    def __init__(self, integration_layer: 'IntegrationLayer'):
        self.integration_layer = integration_layer

    async def validate_attendance(self, employee_id: str, date: str, shift_id: str) -> Tuple[bool, str]:
        """Validate attendance entry against HRIS and shift assignment."""
        # Check employee exists
        employee = await self.integration_layer.call_hris("get_employee", {"employee_id": employee_id})
        if not employee or not employee.get("exists"):
            return False, "ERR_INVALID_USER"
        # Check shift assignment
        assigned = await self.integration_layer.call_hris("check_shift_assignment", {"employee_id": employee_id, "shift_id": shift_id, "date": date})
        if not assigned or not assigned.get("assigned"):
            return False, "ERR_MISSING_ATTENDANCE"
        return True, "VALID"

    async def detect_anomalies(self, date_range: str, department: str) -> List[dict]:
        """Detect irregular attendance patterns."""
        anomalies = await self.integration_layer.call_report_generator("detect_anomalies", {"date_range": date_range, "department": department})
        return anomalies or []

    async def generate_report(self, date_range: str, department: str) -> Dict[str, Any]:
        """Generate attendance report."""
        report = await self.integration_layer.call_report_generator("generate_report", {"date_range": date_range, "department": department})
        return report or {}

    async def validate_shift_assignment(self, employee_id: str, shift_id: str) -> bool:
        """Ensure attendance is only logged for assigned shifts."""
        assigned = await self.integration_layer.call_hris("check_shift_assignment", {"employee_id": employee_id, "shift_id": shift_id})
        return bool(assigned and assigned.get("assigned"))

# =========================
# Integration Layer
# =========================

class IntegrationLayer:
    """Facilitates communication with external systems."""

    def __init__(self):
        self.hris_url = Config.HRIS_API_URL
        self.hris_token = Config.HRIS_API_TOKEN
        self.report_url = Config.REPORT_API_URL
        self.report_key = Config.REPORT_API_KEY
        self.notification_url = Config.NOTIFICATION_API_URL
        self.notification_key = Config.NOTIFICATION_API_KEY

    async def call_hris(self, action: str, params: dict) -> dict:
        """Call HRIS API for employee/shift data."""
        headers = {"Authorization": f"Bearer {self.hris_token}"}
        url = f"{self.hris_url}/{action}"
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=params, timeout=5)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return {}
            except Exception as e:
                logger.error(f"HRIS API error: {e}")
                await asyncio.sleep(exponential_backoff(attempt))
        return {}

    async def call_report_generator(self, action: str, params: dict) -> dict:
        """Call Attendance Report Generator API."""
        headers = {"x-api-key": self.report_key}
        url = f"{self.report_url}/{action}"
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=params, timeout=10)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.error(f"Report Generator API error: {e}")
                await asyncio.sleep(exponential_backoff(attempt))
        return {}

    async def send_notification(self, employee_id: str, message: str) -> dict:
        """Send notification to employee."""
        headers = {"x-api-key": self.notification_key}
        payload = {"employee_id": employee_id, "message": message}
        url = f"{self.notification_url}/notify"
        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=5)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.error(f"Notification API error: {e}")
                await asyncio.sleep(exponential_backoff(attempt))
        return {"delivery_status": "failed"}

# =========================
# Audit Logger
# =========================

class AuditLogger:
    """Logs all access and modification actions, masks PII."""

    def __init__(self):
        self.audit_log = []

    def log_action(self, action: str, user: str, details: dict):
        """Log action with masked PII."""
        entry = {
            "action": action,
            "user": mask_pii(user),
            "details": redact_sensitive(details),
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }
        logger.info(f"Audit log: {entry}")
        self.audit_log.append(entry)

# =========================
# LLM Interaction Manager
# =========================

class LLMInteractionManager:
    """Manages prompt construction, LLM calls, fallback, and formatting."""

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.fallback_model = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.system_prompt = (
            "You are the Healthcare Employee Attendance Tracker, a professional assistant for HR and administrative staff. "
            "Your role is to accurately record, validate, and report employee attendance, ensuring compliance with healthcare policies and regulations. "
            "Always verify user authorization, maintain data privacy, and provide clear, concise responses."
        )
        self.user_prompt_template = (
            "Please specify the employee ID, date, and shift for attendance tracking. For reports, indicate the date range and department."
        )
        self.few_shot_examples = [
            "Record attendance for employee 12345 on 2024-06-10 for morning shift.",
            "Generate attendance report for Cardiology department for the week of 2024-06-03 to 2024-06-09."
        ]

    async def generate_response(self, user_message: str) -> str:
        """Generate LLM response with fallback on failure."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model if attempt == 0 else self.fallback_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM call error (attempt {attempt+1}): {e}")
                await asyncio.sleep(exponential_backoff(attempt))
        return "Sorry, I was unable to process your request at this time. Please contact HR."

# =========================
# Output Formatter
# =========================

class OutputFormatter:
    """Formats responses, applies templates, redacts sensitive data."""

    def __init__(self):
        self.templates = {
            "attendance_success": Template("Attendance for employee {{ employee_id }} on {{ date }} (shift: {{ shift_id }}) has been recorded successfully."),
            "attendance_error": Template("Failed to record attendance: {{ reason }}"),
            "report_success": Template("Attendance report for {{ department }} ({{ date_range }}) is available: {{ report_url }}"),
            "report_error": Template("Failed to generate report: {{ reason }}"),
            "anomaly_detected": Template("Anomalies detected: {{ anomalies }}"),
            "notification_sent": Template("Notification sent to employee {{ employee_id }}."),
            "notification_error": Template("Failed to send notification: {{ reason }}")
        }

    def format_response(self, template_name: str, context: dict) -> str:
        """Format response using template and redact sensitive data."""
        template = self.templates.get(template_name)
        if not template:
            return "Unknown response template."
        context = redact_sensitive(context)
        return template.render(**context)

    def apply_template(self, template_name: str, context: dict) -> str:
        return self.format_response(template_name, context)

# =========================
# Main Agent
# =========================

class AttendanceTrackerAgent(BaseAgent):
    """Healthcare Employee Attendance Tracker Agent."""

    def __init__(self):
        super().__init__()
        self.input_processor = InputProcessor()
        self.auth_service = AuthenticationService()
        self.integration_layer = IntegrationLayer()
        self.domain_logic = DomainLogicEngine(self.integration_layer)
        self.audit_logger = AuditLogger()
        self.llm_manager = LLMInteractionManager()
        self.output_formatter = OutputFormatter()

    async def record_attendance(self, data: dict, user_token: str) -> dict:
        """Log and validate employee attendance entries for each shift."""
        try:
            input_model = self.input_processor.parse_input(data, "attendance")
            if not self.auth_service.authenticate_user(user_token):
                return self._error("ERR_UNAUTHORIZED", "User authentication failed.", "Check your session or login again.")
            if not self.auth_service.authorize_action(input_model.employee_id, "record_attendance"):
                return self._error("ERR_FORBIDDEN", "User not authorized for this action.", "Contact HR for access.")
            valid, code = await self.domain_logic.validate_attendance(input_model.employee_id, input_model.date, input_model.shift_id)
            if not valid:
                reason = "Invalid user." if code == "ERR_INVALID_USER" else "Attendance not allowed for this shift."
                self.audit_logger.log_action("attendance_failed", input_model.employee_id, data)
                return self._error(code, reason, "Check employee ID and shift assignment.")
            # Record attendance in HRIS
            result = await self.integration_layer.call_hris("record_attendance", data)
            if result.get("status") == "success":
                self.audit_logger.log_action("attendance_recorded", input_model.employee_id, data)
                msg = self.output_formatter.format_response("attendance_success", data)
                return {"success": True, "status": "success", "message": msg}
            else:
                self.audit_logger.log_action("attendance_failed", input_model.employee_id, data)
                return self._error("ERR_API_FAILURE", "Failed to record attendance.", "Try again later.")
        except ValidationError as ve:
            return self._error("ERR_INPUT_VALIDATION", str(ve), "Check input fields and formatting.")
        except Exception as e:
            logger.error(f"record_attendance error: {e}")
            return self._error("ERR_INTERNAL", "Internal server error.", "Contact support.")

    async def generate_attendance_report(self, data: dict, user_token: str) -> dict:
        """Create attendance summaries for specified date range and department."""
        try:
            input_model = self.input_processor.parse_input(data, "report")
            if not self.auth_service.authenticate_user(user_token):
                return self._error("ERR_UNAUTHORIZED", "User authentication failed.", "Check your session or login again.")
            if not self.auth_service.authorize_action("report", "generate_attendance_report"):
                return self._error("ERR_FORBIDDEN", "User not authorized for this action.", "Contact HR for access.")
            report = await self.domain_logic.generate_report(input_model.date_range, input_model.department)
            if report.get("report_url"):
                self.audit_logger.log_action("report_generated", "system", data)
                msg = self.output_formatter.format_response("report_success", {
                    "department": input_model.department,
                    "date_range": input_model.date_range,
                    "report_url": report["report_url"]
                })
                return {"success": True, "status": "success", "report_url": report["report_url"], "message": msg}
            else:
                self.audit_logger.log_action("report_failed", "system", data)
                return self._error("ERR_REPORT_FAILURE", "Failed to generate report.", "Try again later.")
        except ValidationError as ve:
            return self._error("ERR_INPUT_VALIDATION", str(ve), "Check input fields and formatting.")
        except Exception as e:
            logger.error(f"generate_attendance_report error: {e}")
            return self._error("ERR_INTERNAL", "Internal server error.", "Contact support.")

    async def detect_anomalies(self, data: dict, user_token: str) -> dict:
        """Identify and flag irregular attendance patterns."""
        try:
            input_model = self.input_processor.parse_input(data, "anomaly")
            if not self.auth_service.authenticate_user(user_token):
                return self._error("ERR_UNAUTHORIZED", "User authentication failed.", "Check your session or login again.")
            if not self.auth_service.authorize_action("anomaly", "detect_anomalies"):
                return self._error("ERR_FORBIDDEN", "User not authorized for this action.", "Contact HR for access.")
            anomalies = await self.domain_logic.detect_anomalies(input_model.date_range, input_model.department)
            self.audit_logger.log_action("anomaly_detection", "system", data)
            msg = self.output_formatter.format_response("anomaly_detected", {"anomalies": anomalies})
            return {"success": True, "status": "success", "anomalies": anomalies, "message": msg}
        except ValidationError as ve:
            return self._error("ERR_INPUT_VALIDATION", str(ve), "Check input fields and formatting.")
        except Exception as e:
            logger.error(f"detect_anomalies error: {e}")
            return self._error("ERR_INTERNAL", "Internal server error.", "Contact support.")

    async def notify_employee(self, data: dict, user_token: str) -> dict:
        """Send notifications for missed or late attendance entries."""
        try:
            input_model = self.input_processor.parse_input(data, "notification")
            if not self.auth_service.authenticate_user(user_token):
                return self._error("ERR_UNAUTHORIZED", "User authentication failed.", "Check your session or login again.")
            if not self.auth_service.authorize_action(input_model.employee_id, "notify_employee"):
                return self._error("ERR_FORBIDDEN", "User not authorized for this action.", "Contact HR for access.")
            result = await self.integration_layer.send_notification(input_model.employee_id, input_model.message)
            if result.get("delivery_status") == "sent":
                self.audit_logger.log_action("notification_sent", input_model.employee_id, data)
                msg = self.output_formatter.format_response("notification_sent", {"employee_id": input_model.employee_id})
                return {"success": True, "status": "sent", "message": msg}
            else:
                self.audit_logger.log_action("notification_failed", input_model.employee_id, data)
                return self._error("ERR_NOTIFICATION_FAILURE", "Failed to send notification.", "Try again later.")
        except ValidationError as ve:
            return self._error("ERR_INPUT_VALIDATION", str(ve), "Check input fields and formatting.")
        except Exception as e:
            logger.error(f"notify_employee error: {e}")
            return self._error("ERR_INTERNAL", "Internal server error.", "Contact support.")

    async def validate_shift_assignment(self, data: dict, user_token: str) -> dict:
        """Ensure attendance is only logged for assigned shifts."""
        try:
            input_model = self.input_processor.parse_input(data, "attendance")
            if not self.auth_service.authenticate_user(user_token):
                return self._error("ERR_UNAUTHORIZED", "User authentication failed.", "Check your session or login again.")
            assigned = await self.domain_logic.validate_shift_assignment(input_model.employee_id, input_model.shift_id)
            if assigned:
                return {"success": True, "status": "valid", "message": "Shift assignment is valid."}
            else:
                return self._error("ERR_MISSING_ATTENDANCE", "Attendance not allowed for this shift.", "Check shift assignment.")
        except ValidationError as ve:
            return self._error("ERR_INPUT_VALIDATION", str(ve), "Check input fields and formatting.")
        except Exception as e:
            logger.error(f"validate_shift_assignment error: {e}")
            return self._error("ERR_INTERNAL", "Internal server error.", "Contact support.")

    async def llm_text_analysis(self, text: str, analysis_type: str) -> dict:
        """Use LLM for language detection, sentiment analysis, entity extraction, text classification."""
        prompt = f"Perform {analysis_type} on the following text:\n\n{text}\n\nReturn the result as JSON."
        try:
            response = await self.llm_manager.generate_response(prompt)
            # Try to parse JSON from LLM response
            try:
                import json
                result = json.loads(response)
                return {"success": True, "result": result}
            except Exception:
                return {"success": True, "result": response}
        except Exception as e:
            logger.error(f"llm_text_analysis error: {e}")
            return self._error("ERR_LLM_FAILURE", "Failed to analyze text.", "Try again later.")

    def _error(self, code: str, message: str, tip: str) -> dict:
        return {
            "success": False,
            "error": {
                "type": code,
                "message": message,
                "tip": tip
            }
        }

# =========================
# FastAPI App & Endpoints
# =========================

app = FastAPI(
    title="Healthcare Employee Attendance Tracker",
    description="API for tracking, validating, and reporting healthcare employee attendance.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AttendanceTrackerAgent()

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "type": "ERR_INPUT_VALIDATION",
                "message": str(exc),
                "tip": "Check your input fields, quotes, commas, and formatting."
            }
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "ERR_HTTP",
                "message": exc.detail,
                "tip": "Check your request and try again."
            }
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "type": "ERR_INTERNAL",
                "message": "Internal server error.",
                "tip": "Check your request or contact support."
            }
        }
    )

@app.middleware("http")
async def json_size_validator(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.body()
            if len(body) > Config.MAX_TEXT_LENGTH:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "success": False,
                        "error": {
                            "type": "ERR_INPUT_TOO_LARGE",
                            "message": f"Input exceeds {Config.MAX_TEXT_LENGTH} bytes.",
                            "tip": "Reduce input size and try again."
                        }
                    }
                )
        except Exception as e:
            logger.error(f"Body read error: {e}")
    return await call_next(request)

@app.post("/attendance/record")
async def record_attendance(request: Request):
    try:
        data = await request.json()
        user_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        return await agent.record_attendance(data, user_token)
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "tip": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
                }
            }
        )

@app.post("/attendance/report")
async def generate_attendance_report(request: Request):
    try:
        data = await request.json()
        user_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        return await agent.generate_attendance_report(data, user_token)
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "tip": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
                }
            }
        )

@app.post("/attendance/anomaly")
async def detect_anomalies(request: Request):
    try:
        data = await request.json()
        user_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        return await agent.detect_anomalies(data, user_token)
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "tip": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
                }
            }
        )

@app.post("/attendance/notify")
async def notify_employee(request: Request):
    try:
        data = await request.json()
        user_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        return await agent.notify_employee(data, user_token)
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "tip": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
                }
            }
        )

@app.post("/attendance/validate-shift")
async def validate_shift_assignment(request: Request):
    try:
        data = await request.json()
        user_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        return await agent.validate_shift_assignment(data, user_token)
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "tip": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
                }
            }
        )

@app.post("/llm/analyze")
async def llm_text_analysis(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        analysis_type = body.get("analysis_type", "sentiment analysis")
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "success": False,
                    "error": {
                        "type": "ERR_INPUT_VALIDATION",
                        "message": "Text input is required.",
                        "tip": "Provide a non-empty 'text' field."
                    }
                }
            )
        if len(text) > Config.MAX_TEXT_LENGTH:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "success": False,
                    "error": {
                        "type": "ERR_INPUT_TOO_LARGE",
                        "message": f"Input exceeds {Config.MAX_TEXT_LENGTH} characters.",
                        "tip": "Reduce input size and try again."
                    }
                }
            )
        return await agent.llm_text_analysis(text, analysis_type)
    except Exception as e:
        logger.error(f"Malformed JSON: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "type": "ERR_MALFORMED_JSON",
                    "message": "Malformed JSON in request body.",
                    "tip": "Ensure your JSON is valid. Check for missing quotes, commas, or brackets."
                }
            }
        )

@app.get("/health")
async def health_check():
    return {"success": True, "status": "ok"}

# =========================
# Main Execution Block
# =========================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Healthcare Employee Attendance Tracker Agent...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)

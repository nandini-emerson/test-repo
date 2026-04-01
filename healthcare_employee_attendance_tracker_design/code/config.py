
# config.py

import os
from dotenv import load_dotenv

class ConfigError(Exception):
    pass

class Config:
    # 1. Environment variable loading
    load_dotenv()

    # 2. API key management
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HRIS_API_URL = os.getenv("HRIS_API_URL")
    HRIS_API_TOKEN = os.getenv("HRIS_API_TOKEN")
    ATTENDANCE_REPORT_API_URL = os.getenv("ATTENDANCE_REPORT_API_URL")
    ATTENDANCE_REPORT_API_KEY = os.getenv("ATTENDANCE_REPORT_API_KEY")
    NOTIFICATION_API_URL = os.getenv("NOTIFICATION_API_URL")
    NOTIFICATION_API_KEY = os.getenv("NOTIFICATION_API_KEY")

    # 3. LLM configuration
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are the Healthcare Employee Attendance Tracker, a professional assistant for HR and administrative staff. "
            "Your role is to accurately record, validate, and report employee attendance, ensuring compliance with healthcare policies and regulations. "
            "Always verify user authorization, maintain data privacy, and provide clear, concise responses."
        ),
        "user_prompt_template": (
            "Please specify the employee ID, date, and shift for attendance tracking. For reports, indicate the date range and department."
        ),
        "few_shot_examples": [
            "Record attendance for employee 12345 on 2024-06-10 for morning shift.",
            "Generate attendance report for Cardiology department for the week of 2024-06-03 to 2024-06-09."
        ]
    }

    # 4. Domain-specific settings
    DOMAIN = "healthcare"
    AGENT_NAME = "Healthcare Employee Attendance Tracker"
    API_REQUIREMENTS = [
        {
            "name": "HRIS_API",
            "type": "external",
            "purpose": "Validate employee credentials, retrieve shift assignments, and record attendance entries.",
            "authentication": "OAuth2 or SSO with multi-factor authentication",
            "rate_limits": "100 requests per minute"
        },
        {
            "name": "Attendance_Report_Generator",
            "type": "external",
            "purpose": "Generate daily, weekly, and monthly attendance reports; export in CSV and PDF.",
            "authentication": "API key or SSO",
            "rate_limits": "60 requests per minute"
        },
        {
            "name": "Notification_Service",
            "type": "external",
            "purpose": "Send attendance notifications to employees and alert HR for anomalies.",
            "authentication": "API key",
            "rate_limits": "200 notifications per minute"
        }
    ]
    DEFAULT_REPORT_FORMAT = "PDF"
    DEFAULT_ATTENDANCE_SHIFT = "morning"
    DEFAULT_DEPARTMENT = "General"

    # 5. Validation and error handling
    @classmethod
    def validate(cls):
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not cls.HRIS_API_URL:
            missing.append("HRIS_API_URL")
        if not cls.HRIS_API_TOKEN:
            missing.append("HRIS_API_TOKEN")
        if not cls.ATTENDANCE_REPORT_API_URL:
            missing.append("ATTENDANCE_REPORT_API_URL")
        if not cls.ATTENDANCE_REPORT_API_KEY:
            missing.append("ATTENDANCE_REPORT_API_KEY")
        if not cls.NOTIFICATION_API_URL:
            missing.append("NOTIFICATION_API_URL")
        if not cls.NOTIFICATION_API_KEY:
            missing.append("NOTIFICATION_API_KEY")
        if missing:
            raise ConfigError(f"Missing required API keys or URLs: {', '.join(missing)}")

    # 6. Default values and fallbacks
    @classmethod
    def get_llm_config(cls):
        return cls.LLM_CONFIG

    @classmethod
    def get_api_url(cls, api_name):
        if api_name == "HRIS_API":
            return cls.HRIS_API_URL
        if api_name == "Attendance_Report_Generator":
            return cls.ATTENDANCE_REPORT_API_URL
        if api_name == "Notification_Service":
            return cls.NOTIFICATION_API_URL
        return None

    @classmethod
    def get_api_key(cls, api_name):
        if api_name == "HRIS_API":
            return cls.HRIS_API_TOKEN
        if api_name == "Attendance_Report_Generator":
            return cls.ATTENDANCE_REPORT_API_KEY
        if api_name == "Notification_Service":
            return cls.NOTIFICATION_API_KEY
        return None

    @classmethod
    def get_default(cls, key):
        defaults = {
            "report_format": cls.DEFAULT_REPORT_FORMAT,
            "attendance_shift": cls.DEFAULT_ATTENDANCE_SHIFT,
            "department": cls.DEFAULT_DEPARTMENT
        }
        return defaults.get(key)

# Validate configuration on import
try:
    Config.validate()
except ConfigError as e:
    # Comment out the next line if you want to suppress errors on import
    # raise
    print(f"Configuration error: {e}")


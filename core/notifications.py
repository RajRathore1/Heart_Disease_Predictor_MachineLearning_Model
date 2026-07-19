"""High-risk email alert for single predictions only (batch uploads can
contain many high-risk rows - one email per row would spam an inbox).
Uses stdlib smtplib/email, no new dependency. Credentials come from
Streamlit secrets, which the client configures themselves after deploy -
the feature stays gracefully disabled until then.
"""

import re
import smtplib
from email.message import EmailMessage

import streamlit as st

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Values still present from the shipped secrets.toml template - treat these
# as "not configured yet" so a half-filled file keeps the feature hidden
# instead of producing send failures.
_PLACEHOLDER_VALUES = {
    "your-email@gmail.com", "your-app-password-here",
    "alerts@example.com", "app-specific-password",
}


def is_configured() -> bool:
    try:
        smtp = st.secrets.get("smtp")
        if not smtp:
            return False
        sender_email = str(smtp.get("sender_email", "")).strip()
        sender_password = str(smtp.get("sender_password", "")).strip()
        if not sender_email or not sender_password:
            return False
        if sender_email in _PLACEHOLDER_VALUES or sender_password in _PLACEHOLDER_VALUES:
            return False
        return is_valid_email(sender_email)
    except Exception:
        return False


def is_valid_email(address: str) -> bool:
    return bool(_EMAIL_RE.match(address.strip()))


def send_high_risk_alert(recipient_email: str, patient_summary: dict, probability: float,
                          model_name: str) -> tuple[bool, str]:
    if not is_configured():
        return False, "Email alerts are not configured yet."
    if not is_valid_email(recipient_email):
        return False, f"'{recipient_email}' doesn't look like a valid email address."

    smtp_cfg = st.secrets["smtp"]

    body_lines = [
        "Heart Disease Predictor - High Risk Alert",
        "",
        f"Model used: {model_name}",
        f"Predicted probability of heart disease: {probability:.1%}",
        "",
        "Patient inputs:",
    ]
    for key, value in patient_summary.items():
        body_lines.append(f"  {key}: {value}")
    body_lines += [
        "",
        "This is an automated statistical risk estimate, not a diagnosis. "
        "Please follow up with appropriate clinical evaluation.",
    ]

    message = EmailMessage()
    message["Subject"] = "Heart Disease Predictor - High Risk Alert"
    message["From"] = smtp_cfg["sender_email"]
    message["To"] = recipient_email
    message.set_content("\n".join(body_lines))

    try:
        with smtplib.SMTP(smtp_cfg["host"], int(smtp_cfg["port"]), timeout=10) as server:
            server.starttls()
            server.login(smtp_cfg["sender_email"], smtp_cfg["sender_password"])
            server.send_message(message)
        return True, "Alert email sent."
    except Exception as exc:
        return False, f"Could not send alert email: {exc}"

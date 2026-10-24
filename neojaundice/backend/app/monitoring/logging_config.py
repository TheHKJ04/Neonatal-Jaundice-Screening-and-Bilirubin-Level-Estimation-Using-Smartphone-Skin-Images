"""Configure JSON-formatted application logs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Format selected log fields as one JSON object per line."""

    ALLOWED_EXTRA_FIELDS = (
        "event",
        "method",
        "route",
        "status_code",
        "duration_ms",
        "exception_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for field in self.ALLOWED_EXTRA_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configure a JSON stream handler without adding duplicates."""

    root_logger = logging.getLogger()

    already_configured = any(
        getattr(handler, "_neojaundice_json_handler", False)
        for handler in root_logger.handlers
    )

    if already_configured:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._neojaundice_json_handler = True  # type: ignore[attr-defined]

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

"""Validation for configured ticket fields."""
from __future__ import annotations

import math
import re
from datetime import date

def validate_dynamic(values, fields, require_all=True):
    """Validate submitted values for configured dynamic fields."""
    if not isinstance(values, dict):
        return ["Dynamic values must be a JSON object"]

    errors = []

    known_fields = {
        field["name"]: field
        for field in fields
    }

    for key in values:
        if key not in known_fields:
            errors.append(f"Unknown dynamic field: {key}")

    for name, field in known_fields.items():
        value = values.get(name)
        is_empty = (
            name not in values
            or value is None
            or (isinstance(value, str) and not value.strip())
        )

        if field["required"] and is_empty and (
            require_all or name in values
        ):
            errors.append(f"{field['label']} is required")
            continue
        if is_empty:
            continue

        label = field["label"]
        field_type = field["type"]

        if field_type == "text":
            if not isinstance(value, str):
                errors.append(f"{label} must be text")
        elif field_type == "dropdown":
            if not isinstance(value, str) or value not in field["options"]:
                errors.append(f"Choose a valid {label}")
        elif field_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                errors.append(f"{label} must be a number")
            else:
                try:
                    numeric_value = float(value)
                    if not math.isfinite(numeric_value):
                        raise ValueError
                except (ValueError, TypeError, OverflowError):
                    errors.append(f"{label} must be a finite number")
        elif field_type == "date":
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                errors.append(f"{label} must be a valid date in YYYY-MM-DD format")
            else:
                try:
                    date.fromisoformat(value)
                except ValueError:
                    errors.append(f"{label} must be a valid date in YYYY-MM-DD format")
        elif field_type == "checkbox":
            if not isinstance(value, bool):
                errors.append(f"{label} must be true or false")
            elif field["required"] and not value:
                errors.append(f"{label} must be checked")

    return errors


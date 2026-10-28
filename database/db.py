"""SQLite configuration, connection, and schema initialization."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("TICKETS_DB", ROOT / "data" / "tickets.db"))

def db():
    """Create database tables and return a SQLite connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'Other',
            priority TEXT NOT NULL DEFAULT 'Medium',
            predicted_solution TEXT NOT NULL DEFAULT '',
            response TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'Open',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            dynamic_values TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS field_definitions (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            type TEXT NOT NULL CHECK (
                type IN ('text', 'dropdown', 'number', 'date', 'checkbox')
            ),
            required INTEGER NOT NULL DEFAULT 0,
            options TEXT NOT NULL DEFAULT '[]',
            active INTEGER NOT NULL DEFAULT 1
        );
        """
    )

    # Hide the old built-in field once on existing databases. Using SQLite's
    # user_version keeps a later custom field with the same name active.
    schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
    if schema_version < 1:
        connection.execute(
            "UPDATE field_definitions SET active = 0 WHERE name = ?",
            ("refund_amount",),
        )
        connection.execute("PRAGMA user_version = 1")

    existing_field = connection.execute(
        "SELECT 1 FROM field_definitions LIMIT 1"
    ).fetchone()

    if not existing_field:
        default_fields = [
            ("order_number", "Order number", "text", 0, []),
            (
                "contact_preference",
                "Preferred contact",
                "dropdown",
                0,
                ["Email", "Phone"],
            ),
            ("purchase_date", "Purchase date", "date", 0, []),
            ("urgent_callback", "Callback requested", "checkbox", 0, []),
        ]

        for name, label, field_type, required, options in default_fields:
            connection.execute(
                """
                INSERT INTO field_definitions
                    (name, label, type, required, options)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    label,
                    field_type,
                    required,
                    json.dumps(options),
                ),
            )

    connection.commit()
    return connection


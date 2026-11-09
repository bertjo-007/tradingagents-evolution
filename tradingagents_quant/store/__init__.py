"""Persistent storage of the research core: one SQLite database, versioned by
``PRAGMA user_version``. No ORM; the schema is plain SQL in ``migrations.py``."""

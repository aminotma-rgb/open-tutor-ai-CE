"""Shared fixtures for evaluation protocol tests (S1–S8)."""

import os
import uuid

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest")
os.environ.setdefault("DEBUG", "true")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.database import Base


@pytest.fixture
def db():
    """Fresh in-memory SQLite DB — reset before every test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def user_id():
    return str(uuid.uuid4())


@pytest.fixture
def support_html():
    return "html-balises-debutant"


@pytest.fixture
def support_sql():
    return "sql-requetes-debutant"


@pytest.fixture
def support_js():
    return "javascript-fonctions"

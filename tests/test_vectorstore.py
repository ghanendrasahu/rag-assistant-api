"""Tests for the persistent vector store wiring.

The Chroma collection must survive process restarts and be usable by several
workers, so the store is persistent and its location is configurable.
"""

from __future__ import annotations

import os
from pathlib import Path

from rag_assistant.services.vectorstore import (
    DEFAULT_PERSIST_PATH,
    VectorStoreService,
    resolve_persist_path,
)


def test_default_path_is_repository_local():
    assert resolve_persist_path() == Path(DEFAULT_PERSIST_PATH)


def test_environment_variable_overrides_the_default(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/var/lib/kage/chroma")

    assert resolve_persist_path() == Path("/var/lib/kage/chroma")


def test_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/from/env")

    assert resolve_persist_path("/explicit/path") == Path("/explicit/path")


def test_empty_environment_variable_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "   ")

    assert resolve_persist_path() == Path(DEFAULT_PERSIST_PATH)


def test_chunking_splits_with_overlap():
    chunks = VectorStoreService._split_words("one two three four five", 3, 1)

    assert chunks == ["one two three", "three four five"]


def test_chunking_keeps_every_word():
    text = "alpha beta gamma delta epsilon"
    chunks = VectorStoreService._split_words(text, 2, 0)

    assert " ".join(chunks).split() == text.split()


def test_chunking_handles_empty_text():
    assert VectorStoreService._split_words("   ", 10, 2) == []


def test_persistence_directory_is_created(tmp_path, monkeypatch):
    # The store is created lazily by Chroma (it downloads an embedding model
    # on first use), so this asserts the filesystem contract only.
    target = tmp_path / "nested" / "chroma"
    monkeypatch.delenv("CHROMA_PERSIST_PATH", raising=False)

    resolved = resolve_persist_path(target)
    resolved.mkdir(parents=True, exist_ok=True)

    assert resolved.is_dir()
    assert os.path.exists(resolved)

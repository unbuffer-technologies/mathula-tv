from contextlib import contextmanager

import pytest

from mathula_tv.autocorrect import (
    AI_RESPONSE_SCHEMA,
    AutocorrectError,
    PgSession,
    clean_user_term,
    curated_hint_auto_applies,
    public_dictionary_candidates,
    stable_id,
    validate_ai_response,
    validate_term_category,
)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def execute(self, query, parameters=()):
        self.queries.append((query, parameters))
        return FakeCursor(self.rows)


class FakeStore:
    def __init__(self, rows):
        self.session = FakeSession(rows)

    @contextmanager
    def connect(self):
        yield self.session


def test_free_text_accepts_new_street_lingo():
    assert clean_user_term("  hoshkhari  ") == "hoshkhari"
    assert clean_user_term("eish\t wena") == "eish wena"


def test_free_text_removes_control_characters():
    assert clean_user_term("new\x00term") == "newterm"


def test_free_text_rejects_empty_and_oversized_values():
    with pytest.raises(AutocorrectError, match="empty"):
        clean_user_term("\x00")
    with pytest.raises(AutocorrectError, match="160"):
        clean_user_term("x" * 161)


def test_term_category_is_constrained():
    assert validate_term_category("Slang") == "slang"
    with pytest.raises(AutocorrectError, match="Unsupported"):
        validate_term_category("secret-personal-category")


def test_postgres_query_translation():
    assert PgSession.translate(
        "SELECT * FROM jobs WHERE job_id = ?"
    ) == "SELECT * FROM jobs WHERE job_id = %s"

    translated = PgSession.translate(
        "INSERT OR IGNORE INTO suppressions(a, b) VALUES (?, ?);"
    )
    assert translated.startswith("INSERT INTO suppressions")
    assert translated.endswith("ON CONFLICT DO NOTHING")
    assert translated.count("%s") == 2


def test_ai_cannot_invent_a_new_candidate():
    request = {
        "spans": [
            {
                "correction_id": "c1",
                "candidates": [
                    {"candidate_id": "known", "replacement_text": "term"}
                ],
            }
        ]
    }
    response = {
        "schema_version": AI_RESPONSE_SCHEMA,
        "decisions": [
            {
                "correction_id": "c1",
                "decision": "ACCEPT_CANDIDATE",
                "candidate_id": "invented",
                "confidence": 0.99,
                "reason": "test",
            }
        ],
    }
    with pytest.raises(AutocorrectError, match="invented"):
        validate_ai_response(request, response)


def test_public_dictionary_entries_are_candidates_not_auto_rules():
    store = FakeStore(
        [
            {
                "term_id": "term-1",
                "heard_text": "skibiri",
                "replacement_text": "skibidi",
                "independent_users": 8,
                "distinct_jobs": 10,
                "term_category": "slang",
                "language": "en-ZA",
            }
        ]
    )
    result = public_dictionary_candidates(store, language="en-ZA")
    assert result[0]["source"] == "public_dictionary"
    assert result[0]["scope"] == "public"
    assert result[0]["replacement_text"] == "skibidi"
    assert "mode" not in result[0]


def test_stable_public_ids_are_deterministic():
    first = stable_id("public_term", "heard", "correct", "en-ZA")
    second = stable_id("public_term", "heard", "correct", "en-ZA")
    assert first == second


def test_curated_hint_rules_auto_apply_by_default():
    assert curated_hint_auto_applies({
        "pattern": r"\bhorse[\s-]+curry\b",
        "replacement": "hoshkhari",
    }) is True
    assert curated_hint_auto_applies({
        "pattern": "ambiguous",
        "replacement": "candidate",
        "auto_apply": False,
    }) is False
    assert curated_hint_auto_applies({
        "pattern": "ambiguous",
        "replacement": "candidate",
        "review_required": True,
    }) is False

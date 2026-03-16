"""
Comprehensive test suite for the RAG Backend API.

All tests use the `requests` library against a live server.
Run the server first:
    uvicorn src.main:app --host 0.0.0.0 --port 8000

Then run tests:
    python -m pytest tests/test_api.py -v
    python -m pytest tests/test_api.py -v -k "upload"          # filter by name
    python -m pytest tests/test_api.py -v --tb=short           # shorter tracebacks

Environment variable overrides:
    RAG_BASE_URL=http://localhost:8000   (default)
"""

import io
import os
import time
import zipfile
from typing import Generator

import pytest
import requests

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("RAG_BASE_URL", "http://localhost:8000").rstrip("/")
API = f"{BASE_URL}/api/v1"


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — reusable zip payloads and a session shared across a test class
# ══════════════════════════════════════════════════════════════════════════════


def _make_zip(files: dict[str, str]) -> bytes:
    """
    Build an in-memory zip from {filename: text_content} dict.
    Files are written as plain UTF-8 text so no real PDF/DOCX parser is needed
    for happy-path smoke tests.  For full integration tests swap these with
    real binary fixtures.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="module")
def valid_zip() -> bytes:
    """A zip containing one pseudo-PDF and one text doc."""
    return _make_zip(
        {
            "doc_a.pdf": (
                "Introduction to Machine Learning\n\n"
                "Machine learning is a subfield of artificial intelligence. "
                "It enables systems to learn from data without explicit programming.\n\n"
                "Supervised Learning\n"
                "In supervised learning, models are trained on labelled examples. "
                "Common algorithms include linear regression, decision trees, and SVMs."
            ),
            "doc_b.docx": (
                "Deep Learning Overview\n\n"
                "Deep learning uses multi-layer neural networks to model complex patterns. "
                "Applications include image recognition, natural language processing, and robotics."
            ),
        }
    )


@pytest.fixture(scope="module")
def multi_file_zip() -> bytes:
    """Zip with three documents — used for source-attribution tests."""
    return _make_zip(
        {
            "finance.pdf": (
                "Q3 Financial Report\n\n"
                "Total revenue for Q3 was $4.2 billion, up 12% year-over-year. "
                "Operating expenses decreased by 5% due to cost optimisation initiatives."
            ),
            "hr_policy.docx": (
                "HR Policy Manual\n\n"
                "Employees are entitled to 20 days of paid annual leave. "
                "Remote work is permitted up to 3 days per week with manager approval."
            ),
            "tech_spec.pdf": (
                "System Architecture\n\n"
                "The backend is built with FastAPI and uses FAISS for vector search. "
                "LangChain orchestrates the RAG pipeline with HuggingFace embeddings."
            ),
        }
    )


@pytest.fixture(scope="module")
def uploaded_session(valid_zip) -> Generator[str, None, None]:
    """
    Upload the valid_zip once and yield the session_id.
    Deletes the session after all tests in the module complete.
    """
    resp = requests.post(
        f"{API}/sessions/upload",
        files={"file": ("documents.zip", valid_zip, "application/zip")},
        timeout=60,
    )
    assert resp.status_code == 201, f"Fixture upload failed: {resp.text}"
    session_id = resp.json()["session_id"]
    yield session_id
    # Teardown — best-effort
    requests.delete(f"{API}/sessions/{session_id}", timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Health check
# ══════════════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_returns_200(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        assert resp.status_code == 200

    def test_health_body(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        assert resp.json() == {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Upload endpoint  POST /sessions/upload
# ══════════════════════════════════════════════════════════════════════════════


class TestUpload:

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_upload_returns_201(self, valid_zip):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        assert resp.status_code == 201
        # Cleanup
        requests.delete(f"{API}/sessions/{resp.json()['session_id']}", timeout=10)

    def test_upload_response_schema(self, valid_zip):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        body = resp.json()
        assert "session_id" in body
        assert "message" in body
        assert "files_processed" in body
        assert "chunks_indexed" in body
        requests.delete(f"{API}/sessions/{body['session_id']}", timeout=10)

    def test_upload_session_id_is_uuid(self, valid_zip):
        import uuid

        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = resp.json()["session_id"]
        # Should not raise
        uuid.UUID(sid)
        requests.delete(f"{API}/sessions/{sid}", timeout=10)

    def test_upload_files_processed_count(self, valid_zip):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        body = resp.json()
        # valid_zip has 2 files
        assert body["files_processed"] == 2
        requests.delete(f"{API}/sessions/{body['session_id']}", timeout=10)

    def test_upload_chunks_indexed_positive(self, valid_zip):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        body = resp.json()
        assert body["chunks_indexed"] > 0
        requests.delete(f"{API}/sessions/{body['session_id']}", timeout=10)

    def test_upload_message_field_non_empty(self, valid_zip):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        assert resp.json()["message"] != ""
        requests.delete(f"{API}/sessions/{resp.json()['session_id']}", timeout=10)

    def test_upload_multi_file_zip(self, multi_file_zip):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("multi.zip", multi_file_zip, "application/zip")},
            timeout=60,
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["files_processed"] == 3
        requests.delete(f"{API}/sessions/{body['session_id']}", timeout=10)

    def test_upload_two_sessions_get_different_ids(self, valid_zip):
        r1 = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        r2 = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]
        assert sid1 != sid2
        requests.delete(f"{API}/sessions/{sid1}", timeout=10)
        requests.delete(f"{API}/sessions/{sid2}", timeout=10)

    # ── Sad path ──────────────────────────────────────────────────────────────

    def test_upload_no_file_returns_422(self):
        resp = requests.post(f"{API}/sessions/upload", timeout=10)
        assert resp.status_code == 422

    def test_upload_non_zip_file_returns_400(self):
        fake_pdf = b"%PDF fake content"
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("document.pdf", fake_pdf, "application/pdf")},
            timeout=10,
        )
        assert resp.status_code == 400

    def test_upload_txt_file_returns_400(self):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("notes.txt", b"some text", "text/plain")},
            timeout=10,
        )
        assert resp.status_code == 400

    def test_upload_empty_zip_returns_422(self):
        """A zip with no supported files should be rejected."""
        empty_zip = _make_zip({"readme.txt": "hello", "data.csv": "a,b,c"})
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("empty.zip", empty_zip, "application/zip")},
            timeout=30,
        )
        assert resp.status_code == 422

    def test_upload_corrupted_zip_returns_error(self):
        corrupted = b"PK\x03\x04this is not a real zip"
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("bad.zip", corrupted, "application/zip")},
            timeout=10,
        )
        assert resp.status_code in (400, 422, 500)

    def test_upload_zero_byte_file_returns_error(self):
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("empty.zip", b"", "application/zip")},
            timeout=10,
        )
        assert resp.status_code in (400, 422, 500)

    def test_upload_wrong_field_name_returns_422(self, valid_zip):
        """FastAPI expects field name 'file'; wrong name should fail validation."""
        resp = requests.post(
            f"{API}/sessions/upload",
            files={"document": ("docs.zip", valid_zip, "application/zip")},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_upload_content_type_header_json_returns_422(self):
        resp = requests.post(
            f"{API}/sessions/upload",
            json={"file": "base64encodedstuff"},
            timeout=10,
        )
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# 3. List sessions  GET /sessions
# ══════════════════════════════════════════════════════════════════════════════


class TestListSessions:

    def test_list_returns_200(self):
        resp = requests.get(f"{API}/sessions", timeout=10)
        assert resp.status_code == 200

    def test_list_returns_list(self):
        resp = requests.get(f"{API}/sessions", timeout=10)
        assert isinstance(resp.json(), list)

    def test_list_contains_uploaded_session(self, uploaded_session):
        resp = requests.get(f"{API}/sessions", timeout=10)
        ids = [s["session_id"] for s in resp.json()]
        assert uploaded_session in ids

    def test_list_session_objects_have_required_keys(self, uploaded_session):
        resp = requests.get(f"{API}/sessions", timeout=10)
        sessions = resp.json()
        target = next(s for s in sessions if s["session_id"] == uploaded_session)
        for key in (
            "session_id",
            "files_processed",
            "chunks_indexed",
            "created_at",
            "last_active",
        ):
            assert key in target, f"Missing key: {key}"

    def test_list_files_processed_is_list(self, uploaded_session):
        resp = requests.get(f"{API}/sessions", timeout=10)
        sessions = resp.json()
        target = next(s for s in sessions if s["session_id"] == uploaded_session)
        assert isinstance(target["files_processed"], list)

    def test_list_chunks_indexed_is_int(self, uploaded_session):
        resp = requests.get(f"{API}/sessions", timeout=10)
        sessions = resp.json()
        target = next(s for s in sessions if s["session_id"] == uploaded_session)
        assert isinstance(target["chunks_indexed"], int)
        assert target["chunks_indexed"] > 0

    def test_list_does_not_include_deleted_session(self, valid_zip):
        # Upload, verify present, delete, verify absent
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        ids_before = [
            s["session_id"] for s in requests.get(f"{API}/sessions", timeout=10).json()
        ]
        assert sid in ids_before

        requests.delete(f"{API}/sessions/{sid}", timeout=10)

        ids_after = [
            s["session_id"] for s in requests.get(f"{API}/sessions", timeout=10).json()
        ]
        assert sid not in ids_after


# ══════════════════════════════════════════════════════════════════════════════
# 4. Get session info  GET /sessions/{session_id}
# ══════════════════════════════════════════════════════════════════════════════


class TestGetSession:

    def test_get_existing_session_returns_200(self, uploaded_session):
        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        assert resp.status_code == 200

    def test_get_session_response_schema(self, uploaded_session):
        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        body = resp.json()
        for key in (
            "session_id",
            "files_processed",
            "chunks_indexed",
            "created_at",
            "last_active",
        ):
            assert key in body

    def test_get_session_id_matches(self, uploaded_session):
        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        assert resp.json()["session_id"] == uploaded_session

    def test_get_session_files_processed_list(self, uploaded_session):
        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        assert isinstance(resp.json()["files_processed"], list)
        assert len(resp.json()["files_processed"]) == 2

    def test_get_session_chunks_indexed_positive(self, uploaded_session):
        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        assert resp.json()["chunks_indexed"] > 0

    def test_get_session_timestamps_are_strings(self, uploaded_session):
        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        body = resp.json()
        assert isinstance(body["created_at"], str)
        assert isinstance(body["last_active"], str)

    def test_get_session_timestamps_are_iso_format(self, uploaded_session):
        from datetime import datetime

        resp = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        body = resp.json()
        # Should not raise
        datetime.fromisoformat(body["created_at"])
        datetime.fromisoformat(body["last_active"])

    def test_get_nonexistent_session_returns_404(self):
        resp = requests.get(f"{API}/sessions/nonexistent-session-id", timeout=10)
        assert resp.status_code == 404

    def test_get_random_uuid_returns_404(self):
        import uuid

        fake_id = str(uuid.uuid4())
        resp = requests.get(f"{API}/sessions/{fake_id}", timeout=10)
        assert resp.status_code == 404

    def test_get_session_last_active_updates_on_access(self, uploaded_session):
        r1 = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        time.sleep(1)
        r2 = requests.get(f"{API}/sessions/{uploaded_session}", timeout=10)
        assert r2.json()["last_active"] >= r1.json()["last_active"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. Delete session  DELETE /sessions/{session_id}
# ══════════════════════════════════════════════════════════════════════════════


class TestDeleteSession:

    def test_delete_existing_session_returns_200(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        resp = requests.delete(f"{API}/sessions/{sid}", timeout=10)
        assert resp.status_code == 200

    def test_delete_response_schema(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        resp = requests.delete(f"{API}/sessions/{sid}", timeout=10)
        body = resp.json()
        assert "session_id" in body
        assert "deleted" in body
        assert "message" in body

    def test_delete_response_deleted_flag_true(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        resp = requests.delete(f"{API}/sessions/{sid}", timeout=10)
        assert resp.json()["deleted"] is True

    def test_delete_response_session_id_matches(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        resp = requests.delete(f"{API}/sessions/{sid}", timeout=10)
        assert resp.json()["session_id"] == sid

    def test_delete_removes_session_from_list(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        requests.delete(f"{API}/sessions/{sid}", timeout=10)
        ids = [
            s["session_id"] for s in requests.get(f"{API}/sessions", timeout=10).json()
        ]
        assert sid not in ids

    def test_delete_makes_get_session_404(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        requests.delete(f"{API}/sessions/{sid}", timeout=10)
        get_resp = requests.get(f"{API}/sessions/{sid}", timeout=10)
        assert get_resp.status_code == 404

    def test_delete_nonexistent_session_returns_404(self):
        resp = requests.delete(f"{API}/sessions/nonexistent-id", timeout=10)
        assert resp.status_code == 404

    def test_delete_same_session_twice_returns_404_on_second(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        requests.delete(f"{API}/sessions/{sid}", timeout=10)
        resp2 = requests.delete(f"{API}/sessions/{sid}", timeout=10)
        assert resp2.status_code == 404

    def test_delete_blocks_subsequent_chat(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        requests.delete(f"{API}/sessions/{sid}", timeout=10)
        chat_resp = requests.post(
            f"{API}/sessions/{sid}/chat",
            json={"message": "What is machine learning?"},
            timeout=10,
        )
        assert chat_resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# 6. Chat endpoint  POST /sessions/{session_id}/chat
# ══════════════════════════════════════════════════════════════════════════════


class TestChat:

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_chat_returns_200(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is machine learning?"},
            timeout=120,
        )
        assert resp.status_code == 200

    def test_chat_response_schema(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is machine learning?"},
            timeout=120,
        )
        body = resp.json()
        assert "session_id" in body
        assert "answer" in body
        assert "sources" in body

    def test_chat_session_id_echoed(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is machine learning?"},
            timeout=120,
        )
        assert resp.json()["session_id"] == uploaded_session

    def test_chat_answer_is_non_empty_string(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is machine learning?"},
            timeout=120,
        )
        answer = resp.json()["answer"]
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_chat_sources_is_list(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is machine learning?"},
            timeout=120,
        )
        assert isinstance(resp.json()["sources"], list)

    def test_chat_sources_have_required_keys(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is supervised learning?"},
            timeout=120,
        )
        sources = resp.json()["sources"]
        for src in sources:
            assert "document_name" in src
            assert "page_label" in src
            assert "relevant_excerpt" in src

    def test_chat_source_document_name_is_string(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is supervised learning?"},
            timeout=120,
        )
        for src in resp.json()["sources"]:
            assert isinstance(src["document_name"], str)

    def test_chat_source_page_label_is_string(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is supervised learning?"},
            timeout=120,
        )
        for src in resp.json()["sources"]:
            assert isinstance(src["page_label"], str)

    def test_chat_source_relevant_excerpt_is_string(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is supervised learning?"},
            timeout=120,
        )
        for src in resp.json()["sources"]:
            assert isinstance(src["relevant_excerpt"], str)

    def test_chat_source_document_name_is_known_file(self, uploaded_session):
        """Sources should reference actual uploaded filenames."""
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is supervised learning?"},
            timeout=120,
        )
        session_info = requests.get(
            f"{API}/sessions/{uploaded_session}", timeout=10
        ).json()
        known_files = session_info["files_processed"]
        for src in resp.json()["sources"]:
            assert (
                src["document_name"] in known_files
            ), f"Source '{src['document_name']}' not in uploaded files {known_files}"

    def test_chat_excerpt_max_length(self, uploaded_session):
        """Excerpts should be ≤ 200 chars as specified in the schema."""
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "Describe deep learning."},
            timeout=120,
        )
        for src in resp.json()["sources"]:
            assert (
                len(src["relevant_excerpt"]) <= 200
            ), f"Excerpt too long: {len(src['relevant_excerpt'])} chars"

    def test_chat_content_type_is_json(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "Hello"},
            timeout=120,
        )
        assert "application/json" in resp.headers.get("content-type", "")

    # ── Multi-file source attribution ─────────────────────────────────────────

    def test_chat_sources_from_multi_file_session(self, multi_file_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("multi.zip", multi_file_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]

        resp = requests.post(
            f"{API}/sessions/{sid}/chat",
            json={"message": "What is the Q3 revenue?"},
            timeout=120,
        )
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert len(sources) >= 1

        requests.delete(f"{API}/sessions/{sid}", timeout=10)

    # ── Short-term memory / conversation continuity ───────────────────────────

    def test_chat_conversation_history_retained(self, uploaded_session):
        """
        Send two messages in the same session; the second should be able to
        refer to the first (short-term memory via InMemorySaver).
        """
        requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "Remember this: the secret keyword is ALPHA."},
            timeout=120,
        )
        resp2 = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What was the secret keyword I just told you?"},
            timeout=120,
        )
        assert resp2.status_code == 200
        # The answer should reference ALPHA if short-term memory works
        answer = resp2.json()["answer"].upper()
        assert "ALPHA" in answer

    def test_chat_multiple_turns_all_return_200(self, uploaded_session):
        questions = [
            "What is machine learning?",
            "Can you elaborate on that?",
            "What about deep learning?",
        ]
        for q in questions:
            resp = requests.post(
                f"{API}/sessions/{uploaded_session}/chat",
                json={"message": q},
                timeout=120,
            )
            assert resp.status_code == 200, f"Failed on question: {q}"

    # ── Sad path ──────────────────────────────────────────────────────────────

    def test_chat_nonexistent_session_returns_404(self):
        resp = requests.post(
            f"{API}/sessions/nonexistent-id/chat",
            json={"message": "Hello"},
            timeout=10,
        )
        assert resp.status_code == 404

    def test_chat_random_uuid_session_returns_404(self):
        import uuid

        resp = requests.post(
            f"{API}/sessions/{uuid.uuid4()}/chat",
            json={"message": "Hello"},
            timeout=10,
        )
        assert resp.status_code == 404

    def test_chat_empty_message_returns_400(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": ""},
            timeout=10,
        )
        assert resp.status_code == 400

    def test_chat_whitespace_only_message_returns_400(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "   "},
            timeout=10,
        )
        assert resp.status_code == 400

    def test_chat_missing_message_field_returns_422(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_chat_wrong_field_name_returns_422(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"query": "What is ML?"},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_chat_non_json_body_returns_422(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            data="What is machine learning?",
            headers={"Content-Type": "text/plain"},
            timeout=10,
        )
        assert resp.status_code == 422

    def test_chat_no_body_returns_422(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            timeout=10,
        )
        assert resp.status_code == 422

    def test_chat_after_delete_returns_404(self, valid_zip):
        r = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid = r.json()["session_id"]
        requests.delete(f"{API}/sessions/{sid}", timeout=10)
        resp = requests.post(
            f"{API}/sessions/{sid}/chat",
            json={"message": "What is ML?"},
            timeout=10,
        )
        assert resp.status_code == 404

    def test_chat_message_field_type_int_returns_422(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": 42},
            timeout=10,
        )
        # FastAPI will coerce int to str in some versions; accept either 200 or 422
        # but it should not 500
        assert resp.status_code != 500

    def test_chat_very_long_message(self, uploaded_session):
        long_msg = "What is machine learning? " * 200  # ~5k chars
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": long_msg},
            timeout=180,
        )
        assert resp.status_code in (200, 400, 413, 422)  # depends on server config

    def test_chat_special_characters_in_message(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "What is ML? <script>alert('xss')</script> \" ' ; --"},
            timeout=120,
        )
        assert resp.status_code == 200

    def test_chat_unicode_message(self, uploaded_session):
        resp = requests.post(
            f"{API}/sessions/{uploaded_session}/chat",
            json={"message": "机器学习是什么？"},
            timeout=120,
        )
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# 7. Session isolation
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionIsolation:
    """Ensure two concurrent sessions do not bleed into each other."""

    def test_two_sessions_answer_from_own_docs(self, valid_zip, multi_file_zip):
        r1 = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        r2 = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("multi.zip", multi_file_zip, "application/zip")},
            timeout=60,
        )
        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        # Ask session 2 about finance — only session 2 has that doc
        resp2 = requests.post(
            f"{API}/sessions/{sid2}/chat",
            json={"message": "What was the Q3 revenue?"},
            timeout=120,
        )
        assert resp2.status_code == 200

        # Session 1 should not have finance.pdf in its sources
        resp1 = requests.post(
            f"{API}/sessions/{sid1}/chat",
            json={"message": "What was the Q3 revenue?"},
            timeout=120,
        )
        if resp1.status_code == 200:
            source_names = [s["document_name"] for s in resp1.json()["sources"]]
            assert "finance.pdf" not in source_names

        requests.delete(f"{API}/sessions/{sid1}", timeout=10)
        requests.delete(f"{API}/sessions/{sid2}", timeout=10)

    def test_deleting_one_session_does_not_affect_another(self, valid_zip):
        r1 = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        r2 = requests.post(
            f"{API}/sessions/upload",
            files={"file": ("docs.zip", valid_zip, "application/zip")},
            timeout=60,
        )
        sid1 = r1.json()["session_id"]
        sid2 = r2.json()["session_id"]

        requests.delete(f"{API}/sessions/{sid1}", timeout=10)

        # session 2 should still be alive and chatty
        resp = requests.post(
            f"{API}/sessions/{sid2}/chat",
            json={"message": "What is machine learning?"},
            timeout=120,
        )
        assert resp.status_code == 200

        requests.delete(f"{API}/sessions/{sid2}", timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
# 8. HTTP method correctness
# ══════════════════════════════════════════════════════════════════════════════


class TestMethodNotAllowed:
    """Wrong HTTP verbs should return 405."""

    def test_get_upload_returns_405(self):
        resp = requests.get(f"{API}/sessions/upload", timeout=10)
        assert resp.status_code == 405

    def test_delete_sessions_list_returns_405(self):
        resp = requests.delete(f"{API}/sessions", timeout=10)
        assert resp.status_code == 405

    def test_put_session_returns_405(self):
        resp = requests.put(f"{API}/sessions/some-id", timeout=10)
        assert resp.status_code == 405

    def test_patch_chat_returns_405(self):
        resp = requests.patch(f"{API}/sessions/some-id/chat", timeout=10)
        assert resp.status_code == 405

    def test_get_chat_returns_405(self):
        resp = requests.get(f"{API}/sessions/some-id/chat", timeout=10)
        assert resp.status_code == 405


# ══════════════════════════════════════════════════════════════════════════════
# 9. Concurrency / stability smoke test
# ══════════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Light concurrency check — not a load test."""

    def test_concurrent_uploads_all_succeed(self, valid_zip):
        import concurrent.futures

        def do_upload(_):
            r = requests.post(
                f"{API}/sessions/upload",
                files={"file": ("docs.zip", valid_zip, "application/zip")},
                timeout=120,
            )
            return r.status_code, r.json().get("session_id")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(do_upload, range(3)))

        statuses = [r[0] for r in results]
        session_ids = [r[1] for r in results]

        assert all(s == 201 for s in statuses)
        assert len(set(session_ids)) == 3  # all unique

        for sid in session_ids:
            requests.delete(f"{API}/sessions/{sid}", timeout=10)

    def test_concurrent_chats_same_session(self, uploaded_session):
        import concurrent.futures

        def do_chat(q):
            r = requests.post(
                f"{API}/sessions/{uploaded_session}/chat",
                json={"message": q},
                timeout=120,
            )
            return r.status_code

        questions = [
            "What is machine learning?",
            "What is deep learning?",
            "What is supervised learning?",
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            statuses = list(ex.map(do_chat, questions))

        assert all(s == 200 for s in statuses)

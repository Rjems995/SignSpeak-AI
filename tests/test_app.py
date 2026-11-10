import json
import re
import pytest
from sqlalchemy import select
from signspeak import create_app, db
from signspeak.models import User, TranslationHistory, PracticeSession, PracticeResult, LearningProgress


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-only-secret", "SQLALCHEMY_DATABASE_URI": "sqlite://", "WTF_CSRF_ENABLED": False, "RATELIMIT_ENABLED": False})
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def register(client, email="one@example.com"):
    return client.post("/register", data={"name": "Robin", "email": email, "password": "a-long-test-password"})


def test_all_pages_render_and_assets_exist(app):
    client = app.test_client()
    for page in ["/", "/dashboard", "/translate", "/learn", "/practice", "/settings", "/capture", "/login", "/register", "/healthz"]:
        response = client.get(page)
        assert response.status_code == 200, page
    register(client)
    for page in ["/history", "/profile"]:
        assert client.get(page).status_code == 200
    for asset in ["css/style.css", "js/app.js", "js/camera.js", "js/recognition.js", "models/classifier.json", "models/evaluation.json", "mark.svg", "icons.svg"]:
        assert client.get("/static/" + asset).status_code == 200, asset


def test_full_alphabet_lessons_and_capabilities(app):
    from signspeak.content import LETTERS
    client = app.test_client()
    assert list(LETTERS) == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    info = client.get("/api/model").get_json()
    assert info["labels"] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert info["motion_labels"] == ["J", "Z"]
    assert len(info["static_labels"]) == 24
    html = client.get("/learn").get_data(as_text=True)
    assert html.count('class="letter-card"') == 26
    assert 'max="26"' in html
    register(client)
    assert client.post("/api/progress", json={"letter": "Z"}).status_code == 200
    assert client.post("/api/practice", json={"results": [{"expected": "J", "predicted": "J"}, {"expected": "Z", "predicted": None}]}).status_code == 201


def test_collection_workshop_labels_and_actual_public_source_counts(app):
    html = app.test_client().get("/capture").get_data(as_text=True)
    for element in ["capture-session", "capture-hand", "capture-lighting", "capture-angle", "capture-verified", "download-static", "download-motion", "coverage-rows"]:
        assert f'id="{element}"' in html
    assert "4770 usable unique static samples" in html
    assert "Q: 199" in html
    assert "Unknown physical hands are not counted as left or right" in html
    assert "5746 usable unique static samples" in html
    assert "Signer IDs: 10" in html
    assert "Review only" in html and "excluded from training" in html
    assert "Lighting benchmark" in html
    assert "Public annotation gaps remain" in html


def test_private_history_and_practice_isolation(app):
    first, second = app.test_client(), app.test_client()
    assert first.get("/api/history").status_code == 401
    assert register(first).status_code == 302
    assert first.post("/api/history", json={"text": "HELLO"}).status_code == 201
    history = first.get("/api/history").get_json()
    assert len(history) == 1
    assert "+00:00" in history[0]["created_at"]
    assert first.post("/api/practice", json={"results": [{"expected": "A", "predicted": "A"}, {"expected": "B", "predicted": None}]}).status_code == 201
    assert register(second, "two@example.com").status_code == 302
    assert second.get("/api/history").get_json() == []
    assert second.delete(f'/api/history/{history[0]["id"]}').status_code == 404
    assert first.get("/dashboard").status_code == 200


def test_validation_history_opt_out_and_search(app):
    client = app.test_client()
    register(client)
    for payload in [[], None, {"text": []}, {"text": " "}, {"text": "a" * 5001}]:
        assert client.post("/api/history", json=payload).status_code == 400
    assert client.post("/api/history", json={"text": "100% private"}).status_code == 201
    assert client.post("/api/history", json={"text": "HELLO"}).status_code == 201
    assert len(client.get("/api/history?q=%25").get_json()) == 1
    assert client.post("/api/settings", json={"save_history": False}).status_code == 200
    assert client.post("/api/history", json={"text": "A"}).status_code == 403
    for payload in [{"threshold": 1.5}, {"rate": "NaN"}, {"theme": []}, {"landmarks": "false"}]:
        assert client.post("/api/settings", json=payload).status_code == 400
    assert client.post("/api/progress", json={"letter": []}).status_code == 400
    assert client.post("/api/practice", json={"results": [{"expected": [], "predicted": "A"}]}).status_code == 400


def test_auth_and_account_deletion_cascade(app):
    client = app.test_client()
    assert client.post("/register", data={"name": "Robin", "email": "broken", "password": "short"}).status_code == 200
    register(client)
    user = db.session.scalar(select(User))
    assert user.password_hash != "a-long-test-password"
    client.post("/api/history", json={"text": "ABC"})
    client.post("/api/progress", json={"letter": "A"})
    client.post("/api/progress", json={"letter": "A"})
    assert len(db.session.scalars(select(LearningProgress)).all()) == 1
    client.post("/api/practice", json={"results": [{"expected": "B", "predicted": "A"}]})
    assert client.post("/api/profile/delete", json={"password": "wrong"}).status_code == 400
    assert client.post("/api/profile/delete", json={"password": "a-long-test-password"}).status_code == 200
    for cls in [User, TranslationHistory, PracticeSession, PracticeResult, LearningProgress]:
        assert db.session.scalars(select(cls)).all() == []
    assert client.get("/api/history").status_code == 401


def test_csrf_required_and_valid_token_works():
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "SQLALCHEMY_DATABASE_URI": "sqlite://", "RATELIMIT_ENABLED": False})
    with app.app_context():
        db.create_all()
    client = app.test_client()
    assert client.post("/register", data={"name": "Test"}).status_code == 400
    html = client.get("/register").get_data(as_text=True)
    token = re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)
    response = client.post("/register", data={"name": "Robin", "email": "csrf@example.com", "password": "valid-long-password", "csrf_token": token})
    assert response.status_code == 302
    assert client.post("/api/history", json={"text": "A"}).status_code == 400
    html = client.get("/translate").get_data(as_text=True)
    token = re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)
    assert client.post("/api/history", json={"text": "A"}, headers={"X-CSRFToken": token}).status_code == 201
    with app.app_context():
        db.drop_all()


import json
import re
from pathlib import Path

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from . import db, limiter
from .content import DEFAULT_SETTINGS, LETTERS
from .models import LearningProgress, PracticeResult, PracticeSession, TranslationHistory, User, UserProfile, UserSettings

web = Blueprint("web", __name__)


def model_info():
    path = Path(current_app.static_folder) / "models" / "metadata.json"
    model = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"ready": False, "labels": []}
    model["static_labels"] = list(model.get("labels", []))
    motion_path = path.with_name("motion-metadata.json")
    motion = json.loads(motion_path.read_text(encoding="utf-8")) if motion_path.exists() else {"ready": False, "labels": []}
    model["motion"] = motion
    model["motion_labels"] = motion.get("labels", []) if motion.get("ready") else []
    model["labels"] = sorted(set(model["static_labels"] + model["motion_labels"]))
    return model


@web.app_context_processor
def context():
    settings = dict(DEFAULT_SETTINGS)
    if current_user.is_authenticated and current_user.settings:
        settings.update(current_user.settings.values)
    return {"letters": LETTERS, "settings": settings, "model": model_info()}


@web.get("/")
def landing():
    return render_template("landing.html", page="home", title="A little gesture. A new connection.")


@web.get("/dashboard")
def dashboard():
    stats = {"translations": 0, "sessions": 0, "accuracy": None, "learned": 0}
    recent = []
    if current_user.is_authenticated:
        uid = current_user.id
        stats["translations"] = db.session.scalar(select(func.count()).select_from(TranslationHistory).where(TranslationHistory.user_id == uid))
        stats["sessions"] = db.session.scalar(select(func.count()).select_from(PracticeSession).where(PracticeSession.user_id == uid))
        results = db.session.scalars(select(PracticeResult).join(PracticeSession).where(PracticeSession.user_id == uid)).all()
        stats["accuracy"] = round(sum(r.correct for r in results) / len(results) * 100) if results else None
        stats["learned"] = db.session.scalar(select(func.count()).select_from(LearningProgress).where(LearningProgress.user_id == uid))
        recent = db.session.scalars(select(TranslationHistory).where(TranslationHistory.user_id == uid).order_by(TranslationHistory.id.desc()).limit(3)).all()
    return render_template("dashboard.html", page="dashboard", title="Your space to connect", stats=stats, recent=recent)


@web.get("/translate")
def translate():
    return render_template("translate.html", page="translate", title="Let your hands do the talking")


@web.get("/learn")
def learn():
    completed = db.session.scalars(select(LearningProgress.letter).where(LearningProgress.user_id == current_user.id)).all() if current_user.is_authenticated else []
    return render_template("learn.html", page="learn", title="Small signs. Meaningful progress.", completed=completed)


@web.get("/practice")
def practice():
    target = request.args.get("letter", "")
    return render_template("practice.html", page="practice", title="A little practice goes a long way", target=target if target in LETTERS else "")


@web.get("/history")
@login_required
def history():
    return render_template("history.html", page="history", title="Pick up the conversation")


@web.get("/settings")
def settings_page():
    return render_template("settings.html", page="settings", title="Make yourself comfortable")


@web.get("/capture")
def capture():
    directory = Path(current_app.static_folder) / "data"
    plan = json.loads((directory / "collection-plan.json").read_text(encoding="utf-8"))
    source_path = directory / "source-coverage.json"
    sources = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else []
    condition_path = directory / "condition-coverage.json"
    condition_coverage = json.loads(condition_path.read_text(encoding="utf-8")) if condition_path.exists() else None
    return render_template("capture.html", page="capture", title="Collect real training samples", collection_plan=plan, collection_sources=sources, condition_coverage=condition_coverage)


@web.get("/profile")
@login_required
def profile():
    return render_template("profile.html", page="profile", title="A space that’s yours")


@web.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute", methods=["POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = db.session.scalar(select(User).where(User.email == email))
        if user and user.check_password(request.form.get("password", "")):
            from flask import session
            session.clear()
            login_user(user)
            return redirect(url_for("web.dashboard"))
        flash("Email or password is incorrect.", "error")
    return render_template("auth.html", page="login", title="Good to see you again", register=False)


@web.route("/register", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or len(name) > 80 or len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email) or not 12 <= len(password) <= 128:
            flash("Enter your name, a valid email, and a password with 12–128 characters.", "error")
        else:
            user = User(email=email, profile=UserProfile(name=name), settings=UserSettings(values=dict(DEFAULT_SETTINGS)))
            user.set_password(password)
            db.session.add(user)
            try:
                db.session.commit()
                from flask import session
                session.clear()
                login_user(user)
                return redirect(url_for("web.dashboard"))
            except IntegrityError:
                db.session.rollback()
                flash("Unable to create an account with these details. Try signing in.", "error")
    return render_template("auth.html", page="register", title="Start with a simple hello", register=True)


@web.post("/logout")
@login_required
def logout():
    logout_user()
    from flask import session
    session.clear()
    return redirect(url_for("web.landing"))


@web.get("/api/model")
def model_status():
    return jsonify(model_info())


@web.route("/api/history", methods=["GET", "POST", "DELETE"])
@login_required
def history_api():
    query = select(TranslationHistory).where(TranslationHistory.user_id == current_user.id)
    if request.method == "GET":
        search = request.args.get("q", "").strip()[:200]
        if search:
            query = query.where(TranslationHistory.text.ilike("%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%", escape="\\"))
        return jsonify([item.to_dict() for item in db.session.scalars(query.order_by(TranslationHistory.id.desc()).limit(100)).all()])
    if request.method == "DELETE":
        for item in db.session.scalars(query).all():
            db.session.delete(item)
        db.session.commit()
        return jsonify(ok=True)
    if not current_user.settings.values.get("save_history", True):
        return jsonify(error="History saving is disabled in your settings."), 403
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 5000:
        return jsonify(error="Enter between 1 and 5,000 characters."), 400
    item = TranslationHistory(user_id=current_user.id, text=text.strip())
    db.session.add(item)
    db.session.commit()
    return jsonify(item.to_dict()), 201


@web.delete("/api/history/<int:item_id>")
@login_required
def delete_history(item_id):
    item = db.session.scalar(select(TranslationHistory).where(TranslationHistory.id == item_id, TranslationHistory.user_id == current_user.id))
    if not item:
        return jsonify(error="Record not found."), 404
    db.session.delete(item)
    db.session.commit()
    return jsonify(ok=True)


@web.post("/api/practice")
@login_required
def save_practice():
    body = request.get_json(silent=True) or {}
    results = body.get("results")
    if not isinstance(results, list) or not 1 <= len(results) <= 60:
        return jsonify(error="A session needs 1–60 attempts."), 400
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("expected"), str) or result.get("expected") not in LETTERS or result.get("predicted") not in [None, *LETTERS]:
            return jsonify(error="Unsupported ASL letter."), 400
    session = PracticeSession(user_id=current_user.id)
    session.results = [PracticeResult(expected=r["expected"], predicted=r.get("predicted"), correct=r["expected"] == r.get("predicted")) for r in results]
    db.session.add(session)
    db.session.commit()
    return jsonify(id=session.id), 201


@web.post("/api/progress")
@login_required
def progress():
    letter = (request.get_json(silent=True) or {}).get("letter")
    if not isinstance(letter, str) or letter not in LETTERS:
        return jsonify(error="Unsupported ASL letter."), 400
    existing = db.session.scalar(select(LearningProgress).where(LearningProgress.user_id == current_user.id, LearningProgress.letter == letter))
    if not existing:
        db.session.add(LearningProgress(user_id=current_user.id, letter=letter))
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
    return jsonify(ok=True)


@web.post("/api/settings")
@login_required
def save_settings():
    body = request.get_json(silent=True) or {}
    values = dict(DEFAULT_SETTINGS)
    try:
        if body.get("theme", "light") not in ["light", "dark"]:
            raise ValueError
        values["theme"] = body.get("theme", "light")
        for key, low, high in [("threshold", 0.6, 0.99), ("hold_ms", 500, 2500), ("rate", 0.5, 2)]:
            val = float(body.get(key, values[key]))
            if not low <= val <= high:
                raise ValueError
            values[key] = val
        for key in ["landmarks", "save_history"]:
            if type(body.get(key, True)) is not bool:
                raise ValueError
            values[key] = body.get(key, True)
        for key in ["camera", "voice"]:
            if not isinstance(body.get(key, ""), str) or len(body.get(key, "")) > 500:
                raise ValueError
            values[key] = body.get(key, "")
    except (TypeError, ValueError):
        return jsonify(error="Some settings are outside the supported range."), 400
    current_user.settings.values = values
    db.session.commit()
    return jsonify(ok=True)


@web.post("/api/profile")
@login_required
def update_profile():
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
        return jsonify(error="Enter a name with 1–80 characters."), 400
    current_user.profile.name = name.strip()
    db.session.commit()
    return jsonify(ok=True)


@web.post("/api/profile/delete")
@login_required
def delete_account():
    password = (request.get_json(silent=True) or {}).get("password", "")
    if not isinstance(password, str) or not current_user.check_password(password):
        return jsonify(error="Your password is incorrect."), 400
    uid = current_user.id
    for cls in [TranslationHistory, LearningProgress, PracticeSession]:
        for item in db.session.scalars(select(cls).where(cls.user_id == uid)).all():
            db.session.delete(item)
    user = db.session.get(User, uid)
    logout_user()
    db.session.delete(user)
    db.session.commit()
    from flask import session
    session.clear()
    return jsonify(ok=True)


@web.get("/healthz")
def health():
    try:
        db.session.execute(select(1))
        return jsonify(status="ok")
    except Exception:
        current_app.logger.exception("Database health check failed")
        return jsonify(status="unavailable"), 503

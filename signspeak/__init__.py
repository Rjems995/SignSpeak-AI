import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFError, CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

db = SQLAlchemy()
login = LoginManager()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=[])


def create_app(test_config=None):
    load_dotenv()
    app = Flask(__name__, instance_relative_config=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    # Persist a development secret without placing it in version control.
    secret_file = Path(app.instance_path) / ".secret"
    secret = os.getenv("SECRET_KEY")
    if not secret:
        if not secret_file.exists():
            secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
        secret = secret_file.read_text(encoding="utf-8").strip()
    uri = os.getenv("DATABASE_URL", "sqlite:///signspeak.db")
    if uri.startswith(("postgres://", "postgresql://")):
        uri = "postgresql+psycopg://" + uri.split("://", 1)[1]
    app.config.update(
        SECRET_KEY=secret, SQLALCHEMY_DATABASE_URI=uri,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
        MAX_CONTENT_LENGTH=128 * 1024,
        RATELIMIT_STORAGE_URI=os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
        WTF_CSRF_TIME_LIMIT=3600,
    )
    if test_config:
        app.config.update(test_config)
    if os.getenv("TRUST_PROXY", "0") == "1":
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    db.init_app(app)
    login.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    login.login_view = "web.login_page"
    from .models import User
    from .routes import web
    app.register_blueprint(web)

    @login.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (ValueError, TypeError):
            return None

    @login.unauthorized_handler
    def unauthorized():
        if request.path.startswith("/api/"):
            return jsonify(error="Sign in to save your progress."), 401
        from flask import redirect, url_for
        return redirect(url_for("web.login_page"))

    @app.errorhandler(CSRFError)
    def csrf_error(error):
        if request.path.startswith("/api/"):
            return jsonify(error="Your session expired. Refresh this page and try again."), 400
        return "Your session expired. Refresh the page and try again.", 400

    @app.before_request
    def validate_json_object():
        if request.path.startswith("/api/") and request.method == "POST":
            if not isinstance(request.get_json(silent=True), dict):
                return jsonify(error="Send a JSON object with the required fields."), 400

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        if request.path.startswith("/api/") or request.endpoint in {"web.profile", "web.history"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.cli.command("init-db")
    def init_db():
        """Create tables in a new database. Use migrations for later schema changes."""
        db.create_all()
        print("Database initialized.")

    return app

